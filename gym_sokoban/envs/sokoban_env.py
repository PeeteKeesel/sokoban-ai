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
        self.penalty_for_step       = -0.1 # reward for making a step
        self.penalty_box_off_target = -1   # reward for pushing box from target
        self.reward_box_on_target   = 1   # reward for pushing a box on a target
        self.reward_finished        = 10  # reward for finishing the game
        self.reward_last            = 0    # reward achieved by the previous step
        self.penalty_already_visited = 0 #-5

        # Other reward types: see 4.2 MCTS configuration
        # self.reward_r0 = 1 if self._check_if_done() else 0

        # The total discounted reward until the current state.
        self.total_return = 0

        # The trajectory of actions taken until the current state.
        self.action_trajectory = []

        # The following is used for IDA*.
        self.g_value = 0  # g(n): the cost to travel from root to node n.
        self.f_value = 0  # f(n) = g(n) + h(n), where h(n) is a problem-
                          # specific heuristic estimate of the cost to travel
                          # from node n to the goal.

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

    def step(self, action, observation_mode='rgb_array', real=True):
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
        if real:
            self.restore_env_states()
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

        self.action_trajectory.append(action)
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }

        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
        if real:
            self.backup_env_states()

        return observation, self.reward_last, done, info

    def steps(self, actions: List[int]):
        """Apply multiple steps in the environment."""
        for a in actions:
            self.step(action=a)

    # TODO: include this in mcts_nnet.py in perform_simulation and set gamma.
    def update_total_return(self, depth, gamma):
        """
        Updates the total return. This should be called in every step of a
        simulation.

        Args:
             depth (int): The current depth of the simulation.

             gamma (float): The discount factor between 0 and 1.
        """
        self.total_return += gamma**depth * self.reward_last

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

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

            return True, box_next_to_player

        return False, False

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
        """Calculates Reward for the previous step."""
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

        for boxPos in boxPositions:
            n  = (boxPos[0] - 1, boxPos[1])
            #nw = (boxPos[0] - 1, boxPos[1] - 1)
            w  = (boxPos[0],     boxPos[1] - 1)
            #sw = (boxPos[0] + 1, boxPos[1] - 1)
            s  = (boxPos[0] + 1, boxPos[1])
            #se = (boxPos[0] + 1, boxPos[1] + 1)
            e  = (boxPos[0],     boxPos[1] + 1)
            #ne = (boxPos[0] - 1, boxPos[1] + 1)
            if (self._is_wall(n) and self._is_wall(w)) \
                or (self._is_wall(w) and self._is_wall(s)) \
                or (self._is_wall(s) and self._is_wall(e)) \
                or (self._is_wall(n) and self._is_wall(e)):
                return True
        return False

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
                end = "" if y < self.room_state.shape[1] - 1 else " "
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

    ###########################################################################
    # Static methods                                                          #
    ###########################################################################

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
    def get_change_coordinates():
        return CHANGE_COORDINATES

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


