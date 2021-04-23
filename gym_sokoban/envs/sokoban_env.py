from gym.utils           import seeding
from gym.spaces.discrete import Discrete
from gym.spaces          import Box
from .room_utils         import generate_room
from .render_utils       import room_to_rgb, room_to_tiny_world_rgb

from copy import deepcopy

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

        #self.searchTree = SearchTree()
        #self.solution = []

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

        self.action_trajectory.append(ACTION_LOOKUP_CHARS[action])

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
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return self.max_steps == self.num_env_steps

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
        self.no_boxes_on_target = 0

        # the positions of the boxes which are not on target states
        #self.boxes_not_on_target = set(tuple(box)  for box  in np.argwhere(self.room_state == 4)[0])
        # the positions of the boxes which are on target states
        #self.boxes_on_target     = set(tuple(box)  for box  in np.argwhere(self.room_state == 3)[0])
        # the positions of the target states for the boxes
        #self.goals               = set(tuple(goal) for goal in np.argwhere(self.room_state == 2)[0])

        # try:
        # Set initial room_state as the root of the search tree.
        #self.searchTree.set_root(self.room_state)
        # except:
        #     print(self.searchTree.get_root())
        #     print("Root was already set")

        starting_observation = self.render(render_mode)
        return starting_observation

    def render_colored(self):
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

    def manhatten_distance(self, pos1, pos2):
        """
        Returns the Manhattan distance between two 2-dimensional points.
        Generally, in a 2d-grid: What is the minimal number of vertical and horizontal
        steps to go to come from position {@pos1} to position {@pos2}.

        Arguments:
            pos1  (2d-list) or (2d-tuple)  - Position in a 2-dimensional plane.
            pos2  (2d-list) or (2d-tuple) - Position in a 2-dimensional plane.
        Returns:
            (float)  - The Manhattan distance between pos1 and pos2.
        """
        assert len(pos1) == len(pos2) == 2
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

        if self._check_if_all_boxes_on_target:  # All boxes are on targets.
            return 0

        # the manhattan distance of the player to the nearest box
        min_dist_player_box = min([self.manhatten_distance(self.player_position, box) for box in boxes_not_on_target])

        # sum of the distances of each box to its nearest goal
        sum_min_dist_boxes_target = sum( min([self.manhatten_distance(target_state, box) for target_state in box_target_states])
                                         for box in boxes_not_on_target )

        return min_dist_player_box + sum_min_dist_boxes_target



    ##############################################################################
    # Get-methods                                                                #
    ##############################################################################

    def get_room_state(self):
        return self.room_state

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
    def get_action_lookup_chars(action):
        if action in ACTION_LOOKUP_CHARS.keys():
            return ACTION_LOOKUP_CHARS[action]
        return None

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
        if self.children_states:
            return [index for index, value in enumerate(self.children_states) if value is not None]
        else:
            # TODO: dont call method inside method.
            return [index for index, value in enumerate(self.get_children()) if value is not None]

    ##############################################################################
    # Search Algorithms                                                          #
    # Those algorithms serve as a comparison to the RL algorithms.               #
    ##############################################################################

    # ----------------------------------------------------------
    # Depth first search algorithm

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

BACKGROUND_COLORS = {"black": "0;{};40",
                     "red": "0;{};41",
                     "green": "0;{};42",
                     "orange": "0;{};43",
                     "blue": "0;{};44",
                     "purple": "0;{};45",
                     "dark green": "0;{};46",
                     "white": "0;{};47"}

COLORS = {"black": "30",
          "red": "31",
          "green": "32",
          "orange": "33",
          "blue": "34",
          "purple": "35",
          "olive green": "36",
          "white": "37"}


def colored_print(text, color, background_color, end=""):
    """
    Prints text with color.
    """
    color_string = BACKGROUND_COLORS[background_color].format(COLORS[color])
    text = f"\x1b[{color_string}m{text}\x1b[0m{end}"
    if end == "":
        print(text, end=end)
    else:
        print(text)

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

BG_COLORS = {
    0: "black",   # wall
    1: "white",   # empty space
    2: "red",     # box target
    3: "blue",    # box on target
    4: "orange",    # box not on target
    5: "green",   # agent
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

CHARS_LOOKUP_ACTIONS = {
    'n': 0,
    'U': 1,
    'D': 2,
    'L': 3,
    'R': 4,
    'u': 5,
    'd': 6,
    'l': 7,
    'r': 8,
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

RENDERING_MODES = ['colored', 'rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw', 'format']
