from gym.utils           import seeding
from gym.spaces.discrete import Discrete
from gym.spaces          import Box
from .room_utils         import generate_room
from .render_utils       import room_to_rgb, room_to_tiny_world_rgb

from copy import deepcopy
from gym_sokoban.envs.room_utils import reverse_move

import gym
import numpy as np


def convert_room_state_to_output_format(mat):
    for key, value in LEVEL_FORMAT.items():
        mat[mat==str(key)] = value
    return mat

def print_room_state(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print(mat[i][j], end='')
        print()
    print()

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
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step       = -0.1 # Neg. reward for making a step
        self.penalty_box_off_target = -1   # Neg. reward for pushing box from target
        self.reward_box_on_target   = 1    # Reward for pushing a box on a target
        self.reward_finished        = 10   # Reward for finishing the game
        self.reward_last            = 0    # Reward achieved by the previous step

        # Other Settings
        self.viewer                 = None
        self.max_steps              = max_steps
        self.action_space           = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space      = Box(low=0,
                                          high=255,
                                          shape=(screen_height, screen_width, 3),
                                          dtype=np.uint8)
        
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

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.

        Arguments:
            action (int): an action of the ACTION_LOOKUP.

        Returns:
             (boolean): indicating a change of the room's state.
        """
        print((action - 1) % 4)
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

    def reverse_step(self, last_action):
        if ACTION_LOOKUP[last_action] != "no operation":
            self.num_env_steps -= 1

            if last_action < 5:
                moved_player, moved_box = self._pull(last_action)
            else:
                pass
            moved_player = False
            moved_box = False

        # no operation was done
        return

    def _pull(self, last_action):
        change = CHANGE_COORDINATES[(last_action + 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        return False, False

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
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

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
        return (self.max_steps == self.num_env_steps)

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
        self.num_env_steps   = 0
        self.reward_last     = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if mode == 'format':
            return self.print_room_state_using_format()

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

    # def _get_all_feasible_actions(self):
    #     feasible_actions = []
    #     print(self.player_position)
    #     for action in range(self.action_space.n):
    #
    #         i_after_action = self.player_position + self.action_equivalent_index_change(action)
    #         if self.room_state[i_after_action] != 0
    #
    #     return feasible_actions
    #
    # def action_equivalent_index_change(self, action):


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

    ##############################################################################
    # Get-methods                                                                #
    ##############################################################################

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    def get_action_by_key(self, key):
        return ACTION_LOOKUP[key]

    def get_player_position(self):
        return self.player_position

    # Find all feasible successor states
    def successors(self):
        succs = []

        for act in range(1, len(ACTION_LOOKUP)):
            state_after_act = self.state_after_action(act)
            #print(f"act={ACTION_LOOKUP[act]}  feasible={state_after_act['state_changed']}  \nnxt_state=\n{state_after_act['new_state']}")
            if state_after_act['state_changed']:
                succs.append(state_after_act['new_state'])
                #print(f"  len(succs)={len(succs)}")

        return succs

    def state_after_action(self, a):
        assert a in ACTION_LOOKUP

        change = CHANGE_COORDINATES[(a-1) % 4]
        cur_pl_pos = self.player_position
        new_pl_pos = cur_pl_pos + change

        if a == 0:  # no operation
            return {'new_state': self.room_state, 'state_changed': False}       # no operation
        if a < 5:   # push operation
            new_box_pos = new_pl_pos + change
            if new_box_pos[0] >= self.room_state.shape[0] or new_box_pos[1] >= self.room_state.shape[1]:
                return {'new_state': self.room_state, 'state_changed': False}   # un-successful push operation

            can_push_box  = self.room_state[tuple(new_pl_pos)]  in [3, 4]
            can_push_box &= self.room_state[tuple(new_box_pos)] in [1, 2]
            if can_push_box:
                new_box_pos, old_box_pos = tuple(new_box_pos), tuple(new_pl_pos)
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_pl_pos)] = 5
                new_room_state[tuple(cur_pl_pos)] = 1

                if self.room_state[new_box_pos] == 2:
                    new_room_state[new_box_pos] = 3     # box on target state
                else:
                    new_room_state[new_box_pos] = 4     # feasible push

                return {'new_state': new_room_state, 'state_changed': True}     # successful push operation
            return {'new_state': self.room_state, 'state_changed': False}       # un-successful push operation
        else:       # move operation
            if self.room_state[tuple(new_pl_pos)] not in [0, 4]:
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_pl_pos)] = 5
                new_room_state[tuple(cur_pl_pos)] = 1

                return {'new_state': new_room_state, 'state_changed': True}     # successful move operation
            else:
                return {'new_state': self.room_state, 'state_changed': False}   # un-successful move operation

    ##############################################################################
    # Search Algorithms                                                          #
    # Those algorithms serve as a comparison to the RL algorithms.               #
    ##############################################################################

    # ----------------------------------------------------------
    # Depth first search algorithm
    def depth_first_search_2(self, print_steps=None):
        """
        @param board: a Board object
        @param print_steps: flag to print intermediate steps
        @return (records, board)
            records: a dictionary keeping track of necessary statistics
            board: a copy of the board at the finished state.
                Contains an array of all moves performed.
        Performs a depth first search on the sokoban board.
        Doesn't add duplicate nodes to the stack so as to prevent looping.
        """
        records = {
            'node' : 0,
            'repeat' : 0,
            'fringe' : 0,
            'explored' : set()
        }

        if print_steps:
            print('repeat\tseen')

        if self._check_if_done(): # board.finished():    # check if initial state is complete
            return records, self

        board_queue = [self]   # initialize queue

        while True:
            if print_steps:
                print("{}\t{}".format(records['repeat'], len(records['explored'])))

            if not board_queue: # if empty queue, fail
                print(records)
                raise Exception('Solution not found.')

            node_board = board_queue.pop(0)
            records['explored'].add(hash(node_board))
            records['fringe'] = len(board_queue)

            if node_board.finished():   # if finished, return
                return records, node_board

            choices = node_board.moves_available()
            if not choices:     # if no options
                board_queue.pop(0)
            else:               # regular
                for direction, cost in choices:
                    records['node'] += 1
                    child_board = deepcopy(node_board).move(direction)

                    if hash(child_board) not in records['explored'] and child_board not in board_queue:
                        board_queue.insert(0, child_board)
                    else:
                        records['repeat'] += 1

    # ----------------------------------------------------------
    # Breadth first search algorithm
    def breadth_first_search(self):
        """TODO"""
        return

    # ----------------------------------------------------------
    # Best first search algorithm
    def best_first_search(self):
        """TODO"""
        return

    # ----------------------------------------------------------
    # A* search algorithm
    def a_star_search(self):
        """TODO"""
        return

    # ----------------------------------------------------------
    # Uniform cost search algorithm (Dijkstra algorithm)
    def uniform_cost_search(self):
        """TODO"""
        return

    ##############################################################################
    # RL Algorithms                                                              #
    ##############################################################################

    # ----------------------------------------------------------
    # Q-Learning
    def q_learning(self):
        """TODO"""
        return

    # ----------------------------------------------------------
    # MCTS
    def monte_carlo_tree_search(self):
        """TODO"""
        return




class Search(gym.Env):
    def __init__(self, env, tree):
        self.env = env
        self.tree = tree

    # Build a tree using DFS. Start from a node (=board-state) and choose actions
    # until either (1) max number of steps achieved or (2) final state is reached.
    def depth_first_search(self, discovered, step, found: bool = False):
        if self.tree.data not in discovered:
            discovered.add(self.tree.data)

            # found the solution
            if self.env._check_if_done():
                found = True
                print(f"Solution found at step {step}")
                return found
            # continue the search
            else:
                self.add_children_nodes()

        return

    def add_children_nodes(self):
        # get all actions possible from the current state
        print(self.env.room_state)
        print(self.env.player_position)
        print(self.env.action_space)

        # 1. Get all possible actions
        # 2. take action & if its not "no operation", add it to the tree
        for action in range(1, len(ACTION_LOOKUP.keys())):
            print(self.env.room_state)

            # take a step
            observation, reward, done, info = self.env.step(action)
            print(info)
            # take the action
            print(action)
            # reverse the action

            # reverse the step




##############################################################################
# Global variables                                                           #
##############################################################################

LEVEL_FORMAT = {
    0: '#',  # wall
    1: ' ',  # empty space
    2: 'T',  # box target
    3: '*',  # box on target
    4: 'B',  # box not on target
    5: '@',  # agent
}

ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

ACTION_LOOKUP_CHARS = {
    0: 'n',
    1: 'U',
    2: 'D',
    3: 'L',
    4: 'R',
    5: 'u',
    6: 'd',
    7: 'l',
    8: 'r',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw', 'format']
