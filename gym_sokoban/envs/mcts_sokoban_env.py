from copy         import deepcopy
from .sokoban_env import SokobanEnv
from src.utils    import *
from typing       import List

import numpy as np


class MctsSokobanEnv(SokobanEnv):

    def __init__(self, dim_room, num_boxes, max_steps=100):

        super(MctsSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, None)

    def get_room_state(self):
        return self.room_state.copy()

    def get_player_position(self):
        return self.player_position

    def get_current_state(self):
        current_state = (self.no_boxes_on_target, self.num_env_steps, self.player_position.copy(), self.room_state.copy())
        return current_state

    def get_children(self):
        """
        Returns a list of the children for the current environment. The index of the list
        represents the action which was taken to get to that child. If the value is None
        the action cannot be taken from the current state.
        """

        children = [None for action in self.get_action_lookup().keys()]

        for action in range(len(self.get_action_lookup())):
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
        assert a in self.get_action_lookup()

        change = self.get_change_coordinates()[(a - 1) % 4]
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
        return [index for index, value in enumerate(self.get_children()) if value is not None]


    def get_obs_for_states(self, states):
        return np.array(states)

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

    def update_total_reward(self):
        #print(f"update_total_reward() called!   reward_last={self.reward_last}   tot_reward={self.total_reward}")
        self.total_reward += self.reward_last

    # TODO: implement get_return correctly
    def get_return(self, state=None, step_idx=None):
        #print(f"get_return() called! total_reward={self.total_reward}")
        return self.total_reward

    def get_best_immediate_action(self, feasible_actions):
        rewards = []
        for action in feasible_actions:
            env_copy = deepcopy(self)
            _, reward, _, _ = env_copy.step(action)
            rewards.append(reward)
        return feasible_actions[np.argmax(rewards)]