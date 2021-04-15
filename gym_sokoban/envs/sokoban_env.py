from typing import Set, List

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

    #def get_root(self):
    #    return self.searchTree.get_root()

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
        A heuristic to estimate the goodness of the board.
        Higher values correspond to a larger 'distances' from the goal state.

        Returns:
            (float)  - the Manhattan distance of the agent to its nearest box plus the sum of all Manhatten distances
                       of each box to its nearest goal state.
        """
        boxes_not_on_target = set(tuple(box) for box in np.argwhere(self.room_state == 4))
        box_target_states   = set(tuple(box) for box in np.argwhere(self.room_state == 2))

        if not boxes_not_on_target:
            return 0

        # the manhattan distance of the player to the nearest box
        min_dist_player_box = min([self.manhatten_distance(self.player_position, box) for box in boxes_not_on_target])

        # sum of the distances of each box to its nearest goal
        sum_min_dist_boxes_target = sum( min([self.manhatten_distance(target_state, box) for target_state in box_target_states]) for box in boxes_not_on_target)

        return min_dist_player_box + sum_min_dist_boxes_target



    ##############################################################################
    # Get-methods                                                                #
    ##############################################################################

    def get_boxes(self):
        pass

    def get_goal_states(self):
        pass

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_player_position(self):
        return self.player_position

    # # Find all feasible successor states
    # def set_children(self):
    #     print("SokobanEnv.set_children()")
    #     succs = [None for a in ACTION_LOOKUP.keys()]
    #     #node = self.searchTree.get_root()
    #
    #     for act in range(1, len(ACTION_LOOKUP)):
    #         state_after_act = self.state_after_action(act)
    #         if state_after_act['state_changed']:
    #             succs[act] = state_after_act['new_state']
    #             self.searchTree.get_root().set_child(room_state=succs[act], reward=0, finished=False, action=act)
    #
    #     return succs
    #
    # # Get the room_state after a given action
    # def state_after_action(self, a):
    #     assert a in ACTION_LOOKUP
    #
    #     change = CHANGE_COORDINATES[(a-1) % 4]
    #     cur_pl_pos = self.player_position
    #     new_pl_pos = cur_pl_pos + change
    #
    #     if a == 0:  # no operation
    #         return {'new_state': self.room_state, 'state_changed': False}       # no operation
    #     if a < 5:   # push operation
    #         new_box_pos = new_pl_pos + change
    #         if new_box_pos[0] >= self.room_state.shape[0] or new_box_pos[1] >= self.room_state.shape[1]:
    #             return {'new_state': self.room_state, 'state_changed': False}   # un-successful push operation
    #
    #         can_push_box  = self.room_state[tuple(new_pl_pos)]  in [3, 4]
    #         can_push_box &= self.room_state[tuple(new_box_pos)] in [1, 2]
    #         if can_push_box:
    #             new_box_pos, old_box_pos = tuple(new_box_pos), tuple(new_pl_pos)
    #             new_room_state = self.room_state.copy()
    #             new_room_state[tuple(new_pl_pos)] = 5
    #             new_room_state[tuple(cur_pl_pos)] = 1
    #
    #             if self.room_state[new_box_pos] == 2:
    #                 new_room_state[new_box_pos] = 3     # box on target state
    #             else:
    #                 new_room_state[new_box_pos] = 4     # feasible push
    #
    #             return {'new_state': new_room_state, 'state_changed': True}     # successful push operation
    #         return {'new_state': self.room_state, 'state_changed': False}       # un-successful push operation
    #     else:       # move operation
    #         if self.room_state[tuple(new_pl_pos)] not in [0, 4]:
    #             new_room_state = self.room_state.copy()
    #             new_room_state[tuple(new_pl_pos)] = 5
    #             new_room_state[tuple(cur_pl_pos)] = 1
    #
    #             return {'new_state': new_room_state, 'state_changed': True}     # successful move operation
    #         else:
    #             return {'new_state': self.room_state, 'state_changed': False}   # un-successful move operation

    def get_children(self):
        """
        Returns a list of the children for the current environment. The index of the list
        represents the action which was taken to get to that child. If the value is None
        the action cannot be taken from the current state.
        """

        children = [None for action in ACTION_LOOKUP.keys()]

        for action in range(1, len(ACTION_LOOKUP)):
            state_after_action = self.state_after_action(action)

            if state_after_action['state_changed']:
                children[action] = state_after_action['new_state']

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

            can_push_box = self.room_state[tuple(new_player_pos)] in [3, 4]
            can_push_box &= self.room_state[tuple(new_box_pos)] in [1, 2]

            if can_push_box:
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_player_pos)] = 5
                new_room_state[tuple(cur_player_pos)] = 1

                if self.room_state[tuple(new_box_pos)] == 2:                    # box on target state
                    new_room_state[tuple(new_box_pos)] = 3
                else:                                                           # feasible push
                    new_room_state[tuple(new_box_pos)] = 4

                return {'new_state': new_room_state, 'state_changed': True}     # successful push operation

            return {'new_state': self.room_state, 'state_changed': False}       # un-successful push operation

        else:
            if self.room_state[tuple(new_player_pos)] not in [0, 4]:
                new_room_state = self.room_state.copy()
                new_room_state[tuple(new_player_pos)] = 5
                new_room_state[tuple(cur_player_pos)] = 1

                return {'new_state': new_room_state, 'state_changed': True}     # successful move operation
            else:
                return {'new_state': self.room_state, 'state_changed': False}   # un-successful move operation

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







# class Search(gym.Env):
#     def __init__(self, env, tree):
#         self.env = env
#         self.tree = tree
#
#     # Build a tree using DFS. Start from a node (=board-state) and choose actions
#     # until either (1) max number of steps achieved or (2) final state is reached.
#     def depth_first_search(self, discovered, step, found: bool = False):
#         if self.tree.data not in discovered:
#             discovered.add(self.tree.data)
#
#             # found the solution
#             if self.env._check_if_done():
#                 found = True
#                 print(f"Solution found at step {step}")
#                 return found
#             # continue the search
#             else:
#                 self.add_children_nodes()
#
#         return
#
#     def add_children_nodes(self):
#         # get all actions possible from the current state
#         print(self.env.room_state)
#         print(self.env.player_position)
#         print(self.env.action_space)
#
#         # 1. Get all possible actions
#         # 2. take action & if its not "no operation", add it to the tree
#         for action in range(1, len(ACTION_LOOKUP.keys())):
#             print(self.env.room_state)
#
#             # take a step
#             observation, reward, done, info = self.env.step(action)
#             print(info)
#             # take the action
#             print(action)
#             # reverse the action
#
#             # reverse the step

##############################################################################
# Additional classes                                                         #
##############################################################################

# this tree serves to save all room_states in a tree

##############################################################################
# SearchTree                                                                 #
##############################################################################
# class SearchTree:
#     def __init__(self):
#         print("__init__ SearchTree called")
#         self.root = None
#
#     def set_root(self, state):
#         #if np.nan(self.root):
#         self.root = SearchTreeNode(state)
#         # else:
#         #     raise ValueError(f"Root is already set\n  it is:{self.root}")
#
#     def get_root(self):
#         return self.root
#
# ##############################################################################
# # SearchTreeNode                                                             #
# ##############################################################################
# class SearchTreeNode:
#     def __init__(self, room_state, reward=0, finished=False, action=None, parent=None, tree=None, env=SokobanEnv):
#         print("__init__ SearchTreeNode called")
#         self.room_state = room_state  # the state of the room
#
#         self.reward     = reward  # the reward for taking the action which led to this node
#         self.finished   = finished  # if this node is a terminal node
#         self.action     = action  # the action which led to this node
#
#         self.tree       = tree  # the tree in which the node is
#         self.parent     = parent  # the parent node
#         self.children   = [None for a in ACTION_LOOKUP.keys()]  # all child nodes
#         self.value      = None  # the current cumulated value
#         self.visits     = 0  # number of times this node was visited
#
#         self.prev_action_traj = []  # the trajectory of actions which led to this node
#
#         self.env = env  # in which environment does the node lie
#
#     # ----------------------------------------------------------------------------
#     # get methods
#     def get_room_state(self):
#         return self.room_state
#
#     def get_children(self):
#         return self.children
#
#     def get_child(self, index):
#         return self.children[index]
#
#     def get_all_feasible_children(self):
#         return [child for child in self.children if child is not None]
#
#     # ----------------------------------------------------------------------------
#     # set methods
#     def set_child(self, room_state, reward, finished, action):
#         assert action in ACTION_LOOKUP
#         print(f"SearchTreeNode.set_child  action={action}")
#
#         env = deepcopy(self.env)
#         env.step(action=action)
#
#         node = SearchTreeNode(room_state=room_state, reward=reward, finished=finished,
#                               action=action, parent=self, tree=self.tree, env=env)
#         self.children[action] = node
#
#         return node
#
# ##############################################################################
# # Searches                                                                   #
# ##############################################################################
# class Searches:
#     def __init__(self, env: SokobanEnv, max_steps: int = 150, n_simulations: int = 10):
#         print("__init__ Searches called")
#         self.env = env  # the Sokoban environment
#         self.n_simulations = n_simulations
#         self.max_steps = max_steps
#         self.tree = SearchTree()  # the tree which consists of the room_states as nodes
#         self.tree.set_root(env.room_state)  # set current room state as the node of the search tree
#
#         self.current_tree = SearchTree()  # the current tree the search is in
#         self.current_tree.set_root(env.room_state)
#         self.current_node = self.current_tree.get_root()
#         self.current_node.env = env
#
#         self.solution = []  # solution trajectory
#
#         self.prev_env = None  # this is need to be in the correct env when using recursion
#         self.current_env = deepcopy(self.current_node.env)  # this is needed to not change the state of the 'real' environment
#
#     def get_children_of_current_trees_root(self):
#         return self.current_tree.get_root().get_children()
#
#     def get_children_of_current_node(self):
#         print("Searches.get_children_of_current_node()")
#         return self.current_node.get_children()
#
#     # Find all feasible successor states
#     def set_children(self):
#         print("Searches.set_children()")
#         succs = [None for a in ACTION_LOOKUP.keys()]
#         # node = self.searchTree.get_root()
#
#         for act in range(1, len(ACTION_LOOKUP)):
#             state_after_act = self.state_after_action(act)
#             if state_after_act['state_changed']:
#                 succs[act] = state_after_act['new_state']
#
#                 self.current_node.set_child(room_state=succs[act], reward=0, finished=False, action=act)
#                 # self.tree.get_root().set_child(room_state=succs[act], reward=0, finished=False, action=act)
#
#         return succs
#
#     # Get the room_state after a given action
#     def state_after_action(self, a):
#         assert a in ACTION_LOOKUP
#         assert np.alltrue(self.current_node.room_state == self.current_node.env.room_state)
#
#         change = CHANGE_COORDINATES[(a - 1) % 4]
#         cur_pl_pos = self.current_node.env.player_position #if self.current_env is not None else self.env.player_position
#         new_pl_pos = cur_pl_pos + change
#
#         if a == 0:  # no operation
#             return {'new_state': self.current_node.room_state, 'state_changed': False}  # no operation
#         if a < 5:  # push operation
#             new_box_pos = new_pl_pos + change
#             if new_box_pos[0] >= self.current_node.room_state.shape[0] or new_box_pos[1] >= self.current_node.room_state.shape[1]:
#                 return {'new_state': self.current_node.room_state, 'state_changed': False}  # un-successful push operation
#
#             can_push_box = self.current_node.room_state[tuple(new_pl_pos)] in [3, 4]
#             can_push_box &= self.current_node.room_state[tuple(new_box_pos)] in [1, 2]
#             if can_push_box:
#                 new_box_pos, old_box_pos = tuple(new_box_pos), tuple(new_pl_pos)
#                 new_room_state = self.current_node.room_state.copy()
#                 new_room_state[tuple(new_pl_pos)] = 5
#                 if self.current_node.parent is not None:
#                     new_room_state[tuple(cur_pl_pos)] = 1 if self.current_node.parent.room_state[tuple(cur_pl_pos)] != 2 else 2
#                 else:
#                     new_room_state[tuple(cur_pl_pos)] = 1
#
#                 if self.current_node.room_state[new_box_pos] == 2:
#                     new_room_state[new_box_pos] = 3  # box on target state
#                 else:
#                     new_room_state[new_box_pos] = 4  # feasible push
#
#                 return {'new_state': new_room_state, 'state_changed': True}  # successful push operation
#             return {'new_state': self.current_node.room_state, 'state_changed': False}  # un-successful push operation
#         else:  # move operation
#             if self.current_node.room_state[tuple(new_pl_pos)] not in [0, 4]:
#                 new_room_state = self.current_node.room_state.copy()
#                 new_room_state[tuple(new_pl_pos)] = 5
#                 if self.current_node.parent is not None:
#                     new_room_state[tuple(cur_pl_pos)] = 1 if self.current_node.parent.room_state[tuple(cur_pl_pos)] != 2 else 2
#                 else:
#                     new_room_state[tuple(cur_pl_pos)] = 1
#
#                 return {'new_state': new_room_state, 'state_changed': True}  # successful move operation
#             else:
#                 return {'new_state': self.current_node.room_state, 'state_changed': False}  # un-successful move operation
#
#     # # Get the room_state after a given action
#     # def state_after_action(self, a):
#     #     assert a in ACTION_LOOKUP
#     #
#     #     change = CHANGE_COORDINATES[(a - 1) % 4]
#     #     cur_pl_pos = self.env.player_position
#     #     new_pl_pos = cur_pl_pos + change
#     #
#     #     if a == 0:  # no operation
#     #         return {'new_state': self.tree.get_root().room_state, 'state_changed': False}  # no operation
#     #     if a < 5:  # push operation
#     #         new_box_pos = new_pl_pos + change
#     #         if new_box_pos[0] >= self.tree.get_root().room_state.shape[0] or new_box_pos[1] >= self.tree.get_root().room_state.shape[1]:
#     #             return {'new_state': self.tree.get_root().room_state, 'state_changed': False}  # un-successful push operation
#     #
#     #         can_push_box = self.tree.get_root().room_state[tuple(new_pl_pos)] in [3, 4]
#     #         can_push_box &= self.tree.get_root().room_state[tuple(new_box_pos)] in [1, 2]
#     #         if can_push_box:
#     #             new_box_pos, old_box_pos = tuple(new_box_pos), tuple(new_pl_pos)
#     #             new_room_state = self.tree.get_root().room_state.copy()
#     #             new_room_state[tuple(new_pl_pos)] = 5
#     #             new_room_state[tuple(cur_pl_pos)] = 1
#     #
#     #             if self.tree.get_root().room_state[new_box_pos] == 2:
#     #                 new_room_state[new_box_pos] = 3  # box on target state
#     #             else:
#     #                 new_room_state[new_box_pos] = 4  # feasible push
#     #
#     #             return {'new_state': new_room_state, 'state_changed': True}  # successful push operation
#     #         return {'new_state': self.tree.get_root().room_state, 'state_changed': False}  # un-successful push operation
#     #     else:  # move operation
#     #         if self.tree.get_root().room_state[tuple(new_pl_pos)] not in [0, 4]:
#     #             new_room_state = self.tree.get_root().room_state.copy()
#     #             new_room_state[tuple(new_pl_pos)] = 5
#     #             new_room_state[tuple(cur_pl_pos)] = 1
#     #
#     #             return {'new_state': new_room_state, 'state_changed': True}  # successful move operation
#     #         else:
#     #             return {'new_state': self.tree.get_root().room_state, 'state_changed': False}  # un-successful move operation
#
#     # ----------------------------------------------------------------------------
#     # search methods
#     def DFS_search(self, discovered: Set, terminated: bool = False, steps:int=0):
#         if discovered is None:
#             discovered = set()
#
#         #if self.current_env is None:
#         #    self.current_env = deepcopy(self.current_node.env) # copy the environment to simulate and therefore not change the actual env
#
#         env = deepcopy(self.current_node.env)
#
#         steps += 1
#         print(f"-----------------------\n>>>>STEPS={steps}\n")
#
#         print("Searches.DFS_search")
#         if self.current_node.parent is not None:
#             print(f"parent.room_state    :\n{self.current_node.parent.room_state}")
#         #print(f"current_env.room_state :\n{self.current_env.room_state}")
#         print(f"current_node.room_state:\n{self.current_node.room_state}")
#
#         # if node is None:
#         #     node = self.tree.get_root()
#
#         # ! the current environments room_state should always be equal to the current_node's room_state
#         # ! Always also change the current_env's state
#         #if np.alltrue(self.current_node.room_state != self.current_env.room_state):
#         #    self.current_env = deepcopy(self.prev_env)
#
#         # # Check if the current nodes's room_state was already discovered before.
#         # print("------- if not np.alltrue(np.in1d(self.current_node.room_state, discovered)) -------")
#         # a = tuple(self.current_node.room_state.flatten())
#         # b = discovered
#         # print(a)
#         # print(b)
#         # print(a in b)
#         # print(np.in1d(a, b))
#         # print(np.alltrue(np.in1d(a, b))) # Falsch, funktioniert nicht
#
#         print("-------------------------------------------------------------------------------------")
#         if not tuple(self.current_node.room_state.flatten()) in discovered:
#             print("self.current_node.room_state is NOT IN discovered")
#
#             # If not, add it to the list of discovered room_states
#             discovered.add(tuple(self.current_node.room_state.flatten()))
#             print(f"discovered = {discovered}")
#
#             # The current node is the terminal node.
#             if self.current_node.env._check_if_done():
#                 terminated = True
#                 self.solution = self.current_node.prev_action_traj
#                 print(f"SOLUTION FOUND :)  solution: {self.solution}")
#                 return terminated
#
#             # TODO: This needs to count in a logical way.
#             elif self.current_node.env.num_env_steps >= self.current_node.env.max_steps:
#                 print(f"MAXIMAL NUMBER OF STEPS = {self.current_node.env.num_env_steps} reached")
#
#             # Todo: ? Include maximal steps in depth of the tree ?
#
#             # Continue the search.
#             else:
#                 # set and get all children SearchTreeNode's to the current SearchTreeNode
#                 self.set_children()
#                 stack = self.current_node.get_children()
#
#                 print(f"len(children): {len(stack)}  len(feasible_children): {len(self.current_node.get_all_feasible_children())}  "
#                       f"\n{self.current_node.room_state}"
#                       f"\n{self.current_node.env.room_state}")
#                 for child in self.current_node.get_all_feasible_children():
#                     print(ACTION_LOOKUP_CHARS[child.action])
#                     print(child.room_state)
#                 print("###########################################################################################")
#
#                 # Search as long as there are successors and none of them is the goal.
#                 while stack \
#                     and not self.current_node.env._check_if_done() \
#                         and not self.current_node.env.num_env_steps >= self.current_node.env.max_steps:
#
#                     nxtNode = stack.pop()
#                     if nxtNode is not None:
#                         print(f"nxtNode.room_state: {nxtNode.room_state}")
#
#                     if (nxtNode is not None) and (not tuple(nxtNode.room_state.flatten()) in discovered):
#                         print(f"    CURRENT NODE inside: "
#                               f"\n{self.current_node.room_state}  "
#                               f"\n{nxtNode.room_state}  {ACTION_LOOKUP_CHARS[len(stack)]}")
#
#                         print(f" get_all_feasible_children: {len(self.current_node.get_all_feasible_children())}")
#                         self.current_node = nxtNode # steps += 1
#                         assert nxtNode.action == self.current_node.action
#
#                         # the previous environment is the current one since we are taking a step here
#                         self.prev_env = deepcopy(self.current_node.env)
#
#                         # append to the action trajectory to remember which actions were taking to get to current_node
#                         if not self.current_node.parent.prev_action_traj:
#                             self.current_node.prev_action_traj.append(ACTION_LOOKUP_CHARS[self.current_node.action])
#                         else:
#                             self.current_node.prev_action_traj = self.current_node.parent.prev_action_traj.copy()
#                             self.current_node.prev_action_traj.append(ACTION_LOOKUP_CHARS[self.current_node.action])
#                         print(f"  prev_action_traj: {self.current_node.prev_action_traj}")
#
#                         print(f"   BEFORE action: {ACTION_LOOKUP_CHARS[self.current_node.action]}\n"
#                               f"{self.prev_env.room_state}\n"
#                               f"{env.room_state}\n"
#                               f"{self.current_node.room_state}")
#
#                         # simulate the step in the temporary environment
#                         env.step(self.current_node.action)
#                         print(f"   AFTER action: {ACTION_LOOKUP_CHARS[self.current_node.action]}\n"
#                               f"{self.prev_env.room_state}\n"
#                               f"{env.room_state}\n"
#                               f"{self.current_node.room_state}")
#
#                         # set the environment of the current_node, which is the nxtNode, to the environment
#                         #   after taking that action.
#                         self.current_node.env = deepcopy(env)
#
#                         print(f"    nxtNode after action '{ACTION_LOOKUP_CHARS[self.current_node.action]}' "
#                               f"\n{self.current_node.room_state}"
#                               f"\nwith the parent "
#                               f"\n{self.current_node.parent.room_state}")
#                         # continue the search in the child next (=nxtNode)
#                         terminated = self.DFS_search(discovered, terminated, steps)
#                     else:
#                         # self.current_env
#                         #print(f"----\nelse: nxtNode is None  for action:{ACTION_LOOKUP[len(stack)]} when room_state is\n{self.current_node.room_state}")
#                         #print(f"     {len(self.current_node.get_all_feasible_children())}")
#                         print(f"else:  {ACTION_LOOKUP_CHARS[len(stack)]}")
#                         if nxtNode is not None:
#                             print(tuple(nxtNode.room_state.flatten()) in discovered)
#
#                     #     print(f"nxtNode.action = {nxtNode.action}")
#                     # # Continue the search from the next Node.
#                     #
#                     # terminated = nxtNode.DFS_search(discovered, terminated)
#
#                     # if not terminated:
#                     #     terminated = False
#
#                 print(f"FOUND={terminated}")
#                 return terminated
#
#         return False
#
#



##############################################################################
# Global methods                                                             #
##############################################################################

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
