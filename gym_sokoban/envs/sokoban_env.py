from copy import deepcopy

from gym.utils           import seeding
from gym.spaces.discrete import Discrete
from gym.spaces          import Box
from .room_utils         import *
from .render_utils       import room_to_rgb, room_to_tiny_world_rgb
from src.utils           import *
from typing              import List

import gym
import numpy as np


class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps is None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.no_boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step       = -0.1 # Neg. reward for making a step
        self.penalty_box_off_target = -1   # Neg. reward for pushing box from target
        self.reward_box_on_target   = 1    # Reward for pushing a box on a target
        self.reward_finished        = 10   # Reward for finishing the game
        self.reward_last            = 0    # Reward achieved by the previous step

        # Other reward types: see 4.2 MCTS configuration
        # self.reward_r0 = 1 if self._check_if_done() else 0

        # TODO: or make this 'return' = discounted reward
        self.total_reward = 0

        self.action_trajectory = []

        # Other Settings
        self.viewer                 = None
        self.max_steps              = max_steps
        self.action_space           = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space      = Box(low=0,
                                          high=255,
                                          shape=(screen_height, screen_width, 3),
                                          dtype=np.uint8)

        self.children_states = []

        # Attributes which will be defined outside __init__
        self.np_random         = None
        self.new_box_position  = None
        self.old_box_position  = None
        self.room_fixed        = None
        self.room_state        = None
        self.box_mapping       = None
        self.player_position   = None
        self.num_env_steps     = None

        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):
        """
        Performs a step in the Sokoban environment.

        Arguments:
            action           (int): an action of the ACTION_LOOKUP.
            observation_mode

         Returns:
            observation (object):  environment-specific object representing
                                   the observation of the environment.
            reward      (float):   amount of reward achieved by the previous
                                   step.
            done        (boolean): whether its time to reset the enviroment.
                                   True, if (1) all boxes are on target
                                         or (2) max. number of steps is reached
                                   False, otherwise
            info        (dict):    diagnostic information useful for debugging.
        """
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False
        # All push actions are in the range of [0, 4]
        elif action < 5:
            moved_player, moved_box = self._push(action)
        else:
            moved_player = self._move(action)

        self._calc_reward()

        # NEW
        #print("=============== step() called ====================")
        self.update_total_reward()
        #print(f"self.total_reward={self.total_reward}")
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }

        self.action_trajectory.append(ACTION_LOOKUP_CHARS[action])

        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    # NEW
    def steps(self, actions: List[int]):
        """Apply multiple steps in the environment."""
        for a in actions:
            self.step(action=a)

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.

        Arguments:
            action (int): an action of the ACTION_LOOKUP.

        Returns:
             (boolean): indicating a change of the room's state.
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False

        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.

        Parameters:
            action (int): the action the player wants to make.

        Returns:
            (boolean): indicating a change of the room's state.
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on

        Returns:
        """
        # Every step a small penalty is given.
        # This ensures that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and
        # give a penalty if a box is pushed off the target.
        if current_boxes_on_target > self.no_boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.no_boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.no_boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        """
        Checks if the game is over. This can either be through
           (1) reaching the maximum number of available steps or
        or (2) by pushing all boxes on the targets.

        Returns:
            (boolean): True, if max steps reached or all boxes
                             got pushed onto the targets,
                       False, otherwise.
        """
        #if self._check_if_maxsteps():
        #    print(f">>>>>>>>   max_steps of {self.max_steps} reached.")
        #elif self._check_if_all_boxes_on_target():
        #    print(f">>>>>>>>   all boxes are on target. after {self.action_trajectory} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        #    self.render_colored()
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return self.max_steps == self.num_env_steps

    def _is_wall(self, pos):
        return self.room_state[pos] == 0

    def _in_corner(self):
        """Checks if any of the boxes on the board is in a corner."""
        boxPositions = np.where(self.room_state == 4)
        boxPositions = list(zip(boxPositions[0], boxPositions[1]))
        #print(f"boxPositions={boxPositions}")

        for boxPos in boxPositions:
            n  = (boxPos[0] - 1, boxPos[1])
            nw = (boxPos[0] - 1, boxPos[1] - 1)
            w  = (boxPos[0],     boxPos[1] - 1)
            sw = (boxPos[0] + 1, boxPos[1] - 1)
            s  = (boxPos[0] + 1, boxPos[1])
            se = (boxPos[0] + 1, boxPos[1] + 1)
            e  = (boxPos[0],     boxPos[1] + 1)
            ne = (boxPos[0] - 1, boxPos[1] + 1)
            if (self._is_wall(n) and self._is_wall(nw) and self._is_wall(nw)) \
                or (self._is_wall(w) and self._is_wall(sw) and self._is_wall(s)) \
                or (self._is_wall(s) and self._is_wall(se) and self._is_wall(e)) \
                or (self._is_wall(n) and self._is_wall(ne) and self._is_wall(e)):
                return True

        return False

    def is_deadlock(self):
        # temp_room_structure = self.room_state.copy()
        # temp_room_structure[temp_room_structure == 5] = 1
        # a, b, c = reverse_move(self.room_state, temp_room_structure,
        #                        self.box_mapping, self.new_box_position, 1)
        # print(self.room_state)
        # print(f"a={a}\n b={b}\n c={c}")

        if self.new_box_position is not None and \
            self.new_box_position not in self.box_mapping and \
            self._in_corner():
            return True
        return False

    def deadlock_detection(self, actionToTake: int):
        """
        Checks if the the state after a taking a given action is a deadlock
        state.

        Arguments:
            actionToTake: int - The action to take from the current state.
        Returns:
            Returns True if the state after {@actionToTake} was taken is a
            deadlock, False if not.
        """
        assert actionToTake in self.action_space

        envAfterAction = deepcopy(self)
        _, rew, _, _ = envAfterAction.step(action=actionToTake)
        # envAfterAction.render_colored()

        if envAfterAction.is_deadlock() and np.abs(envAfterAction.reward_last - self.reward_last) < envAfterAction.reward_box_on_target:
            return True
        return False

    def deadlock_detection_multiple(self, actionsToTake: List[int]):
        """
        Performs a simple deadlock detection for multiple actions and returns
        a list containing if the action results in a deadlock or not.

        Returns:
            List[bool] - List containing True, if the action on the
                         corresponding index in {@actionsToTake} results in a
                         deadlock, False, otherwise.
        """
        return np.array([self.deadlock_detection(a) for a in actionsToTake])


    def get_non_deadlock_feasible_actions(self):
        """Returns all feasible actions excluding deadlock actions."""
        feasible_actions = self.get_feasible_actions()
        deadlocks = self.deadlock_detection_multiple(feasible_actions)
        idxsToRemove = []
        for i, elem in enumerate(feasible_actions):
            if deadlocks[i]:
                idxsToRemove.append(i)

        return np.delete(feasible_actions, idxsToRemove)


    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.no_boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def render_colored(self):
        """
        Render the room state in colored squares to the terminal.
        """
        for x in range(self.room_state.shape[0]):
            for y in range(self.room_state.shape[1]):
                end = "" if y < self.room_state.shape[0] - 1 else " "
                bg_color = BG_COLORS[self.room_state[x][y]]
                color = "white" if bg_color == "black" else "black"
                if self.room_state[x][y] == 5:
                    colored_print(" P ", "red", bg_color, end)
                elif self.room_state[x][y] == 0:
                    colored_print(f"   ", color, bg_color, end)
                else:
                    colored_print(f" {self.room_state[x][y]} ", color, bg_color, end)
        return

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if mode == 'format':
            return self.print_room_state_using_format()

        elif mode == "colored":
            for x in range(self.room_state.shape[0]):
                for y in range(self.room_state.shape[1]):
                    end = "" if y < self.room_state.shape[0] - 1 else " "
                    bg_color = BG_COLORS[self.room_state[x][y]]
                    color = "white" if bg_color == "black" else "black"
                    if self.room_state[x][y] == 5:
                        colored_print(" P ", "red", bg_color, end)
                    elif self.room_state[x][y] == 0:
                        colored_print(f"   ", color, bg_color, end)
                    else:
                        colored_print(f" {self.room_state[x][y]} ", color, bg_color, end)
            return

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def print_room_state_using_format(self):
        print_room_state(convert_room_state_to_output_format(np.copy(self.room_state).astype('str')))

    def manhattan_heuristic(self):
        """
        A heuristic to estimate the cost of the current board.
        Higher values correspond to a larger 'distances' from the goal state.

        Returns:
            (float)  - the Manhattan distance of the agent to its nearest box plus the sum of all Manhatten distances
                       of each box to its nearest goal state.
        """
        boxes_not_on_target = set(tuple(box) for box in np.argwhere(self.room_state == 4))
        box_target_states   = set(tuple(box) for box in np.argwhere(self.room_state == 2))

        if self._check_if_all_boxes_on_target():
            return 0

        # the manhattan distance of the player to the nearest box
        min_dist_player_box = min([manhattan_distance(self.player_position, box) for box in boxes_not_on_target])

        # sum of the distances of each box to its nearest goal
        sum_min_dist_boxes_target = sum( min([manhattan_distance(target_state, box) for target_state in box_target_states])
                                         for box in boxes_not_on_target )

        return min_dist_player_box + sum_min_dist_boxes_target


    ##############################################################################
    # Get-methods                                                                #
    ##############################################################################

    def get_room_state(self):
        return self.room_state.copy()

    def get_current_state(self):
        current_state = (self.no_boxes_on_target, self.num_env_steps, self.player_position.copy(), self.room_state.copy())
        return current_state

    @staticmethod
    def get_n_actions():
        return len(list(ACTION_LOOKUP.keys()))

    @staticmethod
    def get_action_indices():
        return list(ACTION_LOOKUP.keys())

    @staticmethod
    def get_action_lookup():
        return ACTION_LOOKUP

    @staticmethod
    def get_chars_lookup(action):
        return CHARS_LOOKUP_ACTIONS[action]

    @staticmethod
    def get_action_lookup_chars(action: int):
        if action in ACTION_LOOKUP_CHARS.keys():
            return ACTION_LOOKUP_CHARS[action]
        return None

    @staticmethod
    def get_actions_lookup_chars(actions: List[int]) -> List[str]:
        actionsAsChars = []
        for action in actions:
            assert action in ACTION_LOOKUP.keys()
            actionsAsChars.append(ACTION_LOOKUP_CHARS[action])
        return actionsAsChars

    @staticmethod
    def print_actions_as_chars(actions: List[int]):
        if actions:
            actionsAsChars = SokobanEnv.get_actions_lookup_chars(actions)
            return ''.join(actionsAsChars)
        return ''

    def get_player_position(self):
        return self.player_position

    def get_children(self):
        """
        Returns a list of the children for the current environment. The index of the list
        represents the action which was taken to get to that child. If the value is None
        the action cannot be taken from the current state.
        """

        children = [None for action in ACTION_LOOKUP.keys()]

        for action in range(len(ACTION_LOOKUP)):
            state_after_action = self.state_after_action(action)

            if state_after_action['state_changed']:
                children[action] = state_after_action['new_state']

        self.children_states = children
        # print(f"        self.children_states = {np.where(np.array([i is not None for i in self.children_states]) == True)[0]} ")

        return children

    def state_after_action(self, a):
        """
        Returns a dictionary with information about if the state after the given {@action} was changed
        and what the state is. Stays the same state if the action is not feasible.
        """
        assert a in ACTION_LOOKUP

        change = CHANGE_COORDINATES[(a - 1) % 4]
        cur_player_pos = self.player_position
        new_player_pos = cur_player_pos + change

        if a == 0:
            return {'new_state': self.room_state, 'state_changed': False}

        if a < 5:
            new_box_pos = new_player_pos + change

            if new_box_pos[0] >= self.room_state.shape[0] or new_box_pos[1] >= self.room_state.shape[1]:
                return {'new_state': self.room_state, 'state_changed': False}

            can_push_box  = self.room_state[tuple(new_player_pos)] in [3, 4]
            can_push_box &= self.room_state[tuple(new_box_pos)]    in [1, 2]

            if can_push_box:
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_player_pos)] = 5
                new_room_state[tuple(cur_player_pos)] = 1

                if self.room_state[tuple(new_box_pos)] == 2:                 # box on target state
                    new_room_state[tuple(new_box_pos)] = 3
                else:                                                        # feasible push
                    new_room_state[tuple(new_box_pos)] = 4

                return {'new_state': new_room_state, 'state_changed': True}  # successful push operation

            return {'new_state': self.room_state, 'state_changed': False}    # un-successful push operation

        else:
            if self.room_state[tuple(new_player_pos)] not in [0, 4]:
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_player_pos)] = 5
                new_room_state[tuple(cur_player_pos)] = 1

                return {'new_state': new_room_state, 'state_changed': True}     # successful move operation
            else:
                return {'new_state': self.room_state, 'state_changed': False}   # un-successful move operation

    def get_feasible_actions(self):
        """Returns the indices of all feasible actions from the state."""
        # if self.children_states:
        #     return [index for index, value in enumerate(self.children_states) if value is not None]
        # else:
        #     return [index for index, value in enumerate(self.get_children()) if value is not None]
        return [index for index, value in enumerate(self.get_children()) if value is not None]

    # NEW
    def get_obs_for_states(self, states):
        return np.array(states)

    # NEW
    def update_total_reward(self):
        #print(f"update_total_reward() called!   reward_last={self.reward_last}   tot_reward={self.total_reward}")
        self.total_reward += self.reward_last

    # NEW
    # TODO: implement this correctly. This is now just the total reward
    def get_return(self, state=None, step_idx=None):
        #print(f"get_return() called! total_reward={self.total_reward}")
        return self.total_reward