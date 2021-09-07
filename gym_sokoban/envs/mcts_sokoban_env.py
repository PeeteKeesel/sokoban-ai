from copy import deepcopy
from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
from .room_utils import generate_room
from src.utils import *
from typing import List


class MctsSokobanEnv(SokobanEnv):
    """
    Extension of the SokobanEnv class containing methods specifically for MCTS.
    """

    def __init__(self, dim_room, num_boxes, max_steps=100, original_map = None):
        self.original_map = None
        if original_map:
            self.original_map = original_map
        print("HALLO")
        super(MctsSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, None)

    def reset(self, second_player=False, render_mode='rgb_array'):
        # A manual board was given from the user.
        if self.original_map:
            self.room_fixed, self.room_state, self.box_mapping = \
                self.generate_room_from_manual_map(
                    self.original_map
                )

            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = 0
            self.reward_last = 0
            self.no_boxes_on_target = 0
            starting_observation = room_to_rgb(self.room_state, self.room_fixed)
            self.backup_env_states()

            return starting_observation
        # A board will be rendered.
        else:
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

    def generate_room_from_manual_map(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                # wall
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                # player position
                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)

                # box
                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)
                # storage
                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                # empty space
                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping

    def backup_env_states(self):
        self.reward_last_backup = self.reward_last
        self.no_boxes_on_target_backup = self.no_boxes_on_target
        self.num_env_steps_backup = self.num_env_steps
        self.player_position_backup = self.player_position.copy()
        self.room_state_backup = self.room_state.copy()

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

        children = [None for _ in self.get_action_lookup().keys()]

        for action in self.get_action_lookup().keys():
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
            else:
                # Push operation but no box in front of the player.
                # Thus, the player just moves. The new position can also be on
                # a goal position.
                if self.room_state[tuple(new_player_pos)] in [1, 2]:
                    new_room_state = self.room_state.copy()
                    new_room_state[tuple(new_player_pos)] = 5
                    new_room_state[tuple(cur_player_pos)] = 1

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

    def get_children_environments(self):
        feasible_actions = self.get_non_deadlock_feasible_actions()
        children_envs = []
        for feasible_action in feasible_actions:
            env_copy = deepcopy(self)
            env_copy.step(feasible_action)
            children_envs.append(env_copy)
        return children_envs


    def print_room_state_using_format(self):
        print_room_state(convert_room_state_to_output_format(np.copy(self.room_state).astype('str')))

    def get_boxes_not_on_target(self):
        return set(tuple(box) for box in np.argwhere(self.room_state == 4))

    def get_goal_states(self):
        return set(tuple(box) for box in np.argwhere(self.room_state == 2))

    def manhattan_heuristic(self):
        """
        A heuristic to estimate the cost of the current board.
        Higher values correspond to a larger 'distances' from the goal state.

        Returns:
            (float): The Manhattan distance of the agent to its nearest box plus the sum of all Manhatten distances
                     of each box to its nearest goal state.
        """
        boxes_not_on_target = self.get_boxes_not_on_target()
        box_target_states   = self.get_goal_states()

        # The player stands on a goal state of a box.
        if len(box_target_states) < len(boxes_not_on_target):
            box_target_states.add(tuple(np.argwhere(self.room_state == 5)[0]))

        if self._check_if_all_boxes_on_target():
            return 0

        # the manhattan distance of the player to the nearest box
        min_dist_player_box = min([manhattan_distance(self.player_position, box) for box in boxes_not_on_target])
        #print(f"   min_dist_player_box: {min_dist_player_box}")

        # sum of the distances of each box to its nearest goal
        sum_min_dist_boxes_target = sum( min([manhattan_distance(target_state, box) for target_state in box_target_states])
                                         for box in boxes_not_on_target )
        #print(f"   sum_min_dist_boxes_target: {sum_min_dist_boxes_target}")

        return min_dist_player_box + sum_min_dist_boxes_target

    def hungarian_heuristic(self):
        node = self.room_state
        frontier_goals = []
        frontier_boxes = []
        hungarian_table = []
        for row in range(0, node.shape[0]):
            for column in range(0, node.shape[1]):
                if node[row][column] == 2 or node[row][column] == 3:
                    frontier_goals.append(row)
                    frontier_goals.append(column)
                if node[row][column] == 4 or node[row][column] == 3:
                    frontier_boxes.append(row)
                    frontier_boxes.append(column)
        for box_idx in range(0, len(frontier_boxes), 2):
            temp = []
            for goal_idx in range(0, len(frontier_goals), 2):
                distance_x = abs(int(frontier_boxes[int(box_idx)]) - int(frontier_goals[int(goal_idx)]))
                distance_y = abs(int(frontier_boxes[int(box_idx) + 1]) - int(frontier_goals[int(goal_idx) + 1]))
                temp.append(distance_x + distance_y)
            hungarian_table.append(temp)
        #print(hungarian_table)
        hungarian = Hungarian()
        hungarian.calculate(hungarian_table)
        return hungarian.get_total_potential()


    def all_boxes_on_target(self):
        return self._check_if_all_boxes_on_target()

    def max_steps_reached(self):
        return self._check_if_maxsteps()

    def is_done(self):
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def is_deadlock(self):
        if self.new_box_position is not None and \
            self.new_box_position not in self.box_mapping and \
            self._in_corner():
            return True
        return False

    def deadlock_detection(self, actionToTake: int):
        """
        Checks if the the state after a taking a given action is a deadlock
        state.

        Args:
            actionToTake: int - The action to take from the current state.
        Returns:
            Returns True if the state after {@actionToTake} was taken is a
            deadlock, False if not.
        """
        assert actionToTake in self.action_space

        envAfterAction = deepcopy(self)
        _, rew, _, _ = envAfterAction.step(action=actionToTake)

        if envAfterAction.is_deadlock() and \
                np.abs(envAfterAction.reward_last - self.reward_last) < envAfterAction.reward_box_on_target:
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
        self.total_reward += self.reward_last

    def get_total_reward(self):
        return self.total_reward

    def get_return(self):
        return self.total_return

    def get_best_immediate_action(self, feasible_actions):
        rewards = []
        for action in feasible_actions:
            env_copy = deepcopy(self)
            _, reward, _, _ = env_copy.step(action)
            rewards.append(reward)
        return feasible_actions[np.argmax(rewards)]

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states)




########################################################################################################################

class HungarianError(Exception):
    pass

# Import numpy. Error if fails
try:
    import numpy as np
except ImportError:
    raise HungarianError("NumPy is not installed.")


class Hungarian:
    """
    Implementation of the Hungarian (Munkres) Algorithm using np.
    Usage:
        hungarian = Hungarian(cost_matrix)
        hungarian.calculate()
    or
        hungarian = Hungarian()
        hungarian.calculate(cost_matrix)
    Handle Profit matrix:
        hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    or
        cost_matrix = Hungarian.make_cost_matrix(profit_matrix)
    The matrix will be automatically padded if it is not square.
    For that numpy's resize function is used, which automatically adds 0's to any row/column that is added
    Get results and total potential after calculation:
        hungarian.get_results()
        hungarian.get_total_potential()
    """

    def __init__(self, input_matrix=None, is_profit_matrix=False):
        """
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        if input_matrix is not None:
            # Save input
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxColumn = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            # Adds 0s if any columns/rows are added. Otherwise stays unaltered
            matrix_size = max(self._maxColumn, self._maxRow)
            pad_columns = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxColumn
            my_matrix = np.pad(my_matrix, ((0,pad_columns),(0,pad_rows)), 'constant', constant_values=(0))

            # Convert matrix to profit matrix if necessary
            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)

            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            # Results from algorithm.
            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def get_results(self):
        """Get results after calculation."""
        return self._results

    def get_total_potential(self):
        """Returns expected value after calculation."""
        return self._totalPotential

    def calculate(self, input_matrix=None, is_profit_matrix=False):
        """
        Implementation of the Hungarian (Munkres) Algorithm.
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        # Handle invalid and new matrix inputs.
        if input_matrix is None and self._cost_matrix is None:
            raise HungarianError("Invalid input")
        elif input_matrix is not None:
            self.__init__(input_matrix, is_profit_matrix)

        result_matrix = self._cost_matrix.copy()

        # Step 1: Subtract row mins from each row.
        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        # Step 2: Subtract column mins from each column.
        for index, column in enumerate(result_matrix.T):
            result_matrix[:, index] -= column.min()

        # Step 3: Use minimum number of lines to cover all zeros in the matrix.
        # If the total covered rows+columns is not equal to the matrix size then adjust matrix and repeat.
        total_covered = 0
        while total_covered < self._size:
            # Find minimum number of lines to cover all zeros in the matrix and find total covered rows and columns.
            cover_zeros = CoverZeros(result_matrix)
            covered_rows = cover_zeros.get_covered_rows()
            covered_columns = cover_zeros.get_covered_columns()
            total_covered = len(covered_rows) + len(covered_columns)

            # if the total covered rows+columns is not equal to the matrix size then adjust it by min uncovered num (m).
            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_columns)

        # Step 4: Starting with the top row, work your way downwards as you make assignments.
        # Find single zeros in rows or columns.
        # Add them to final result and remove them and their associated row/column from the matrix.
        expected_results = min(self._maxColumn, self._maxRow)
        zero_locations = (result_matrix == 0)
        while len(self._results) != expected_results:

            # If number of zeros in the matrix is zero before finding all the results then an error has occurred.
            if not zero_locations.any():
                raise HungarianError("Unable to find results. Algorithm has failed.")

            # Find results and mark rows and columns for deletion
            matched_rows, matched_columns = self.__find_matches(zero_locations)

            # Make arbitrary selection
            total_matched = len(matched_rows) + len(matched_columns)
            if total_matched == 0:
                matched_rows, matched_columns = self.select_arbitrary_match(zero_locations)

            # Delete rows and columns
            for row in matched_rows:
                zero_locations[row] = False
            for column in matched_columns:
                zero_locations[:, column] = False

            # Save Results
            self.__set_results(zip(matched_rows, matched_columns))

        # Calculate total potential
        value = 0
        for row, column in self._results:
            value += self._input_matrix[row, column]
        self._totalPotential = value

    @staticmethod
    def make_cost_matrix(profit_matrix):
        """
        Converts a profit matrix into a cost matrix.
        Expects NumPy objects as input.
        """
        # subtract profit matrix from a matrix made of the max value of the profit matrix
        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix
        return cost_matrix

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_columns):
        """Subtract m from every uncovered number and add m to every element covered with two lines."""
        # Calculate minimum uncovered number (m)
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        elements.append(element)
        min_uncovered_num = min(elements)

        # Add m to every covered element
        adjusted_matrix = result_matrix
        for row in covered_rows:
            adjusted_matrix[row] += min_uncovered_num
        for column in covered_columns:
            adjusted_matrix[:, column] += min_uncovered_num

        # Subtract m from every element
        m_matrix = np.ones(self._shape, dtype=int) * min_uncovered_num
        adjusted_matrix -= m_matrix

        return adjusted_matrix

    def __find_matches(self, zero_locations):
        """Returns rows and columns with matches in them."""
        marked_rows = np.array([], dtype=int)
        marked_columns = np.array([], dtype=int)

        # Mark rows and columns with matches
        # Iterate over rows
        for index, row in enumerate(zero_locations):
            row_index = np.array([index])
            if np.sum(row) == 1:
                column_index, = np.where(row)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        # Iterate over columns
        for index, column in enumerate(zero_locations.T):
            column_index = np.array([index])
            if np.sum(column) == 1:
                row_index, = np.where(column)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        return marked_rows, marked_columns

    @staticmethod
    def __mark_rows_and_columns(marked_rows, marked_columns, row_index, column_index):
        """Check if column or row is marked. If not marked then mark it."""
        new_marked_rows = marked_rows
        new_marked_columns = marked_columns
        if not (marked_rows == row_index).any() and not (marked_columns == column_index).any():
            new_marked_rows = np.insert(marked_rows, len(marked_rows), row_index)
            new_marked_columns = np.insert(marked_columns, len(marked_columns), column_index)
        return new_marked_rows, new_marked_columns

    @staticmethod
    def select_arbitrary_match(zero_locations):
        """Selects row column combination with minimum number of zeros in it."""
        # Count number of zeros in row and column combinations
        rows, columns = np.where(zero_locations)
        zero_count = []
        for index, row in enumerate(rows):
            total_zeros = np.sum(zero_locations[row]) + np.sum(zero_locations[:, columns[index]])
            zero_count.append(total_zeros)

        # Get the row column combination with the minimum number of zeros.
        indices = zero_count.index(min(zero_count))
        row = np.array([rows[indices]])
        column = np.array([columns[indices]])

        return row, column

    def __set_results(self, result_lists):
        """Set results during calculation."""
        # Check if results values are out of bound from input matrix (because of matrix being padded).
        # Add results to results list.
        for result in result_lists:
            row, column = result
            if row < self._maxRow and column < self._maxColumn:
                new_result = (int(row), int(column))
                self._results.append(new_result)


class CoverZeros:
    """
    Use minimum number of lines to cover all zeros in the matrix.
    Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
    """

    def __init__(self, matrix):
        """
        Input a matrix and save it as a boolean matrix to designate zero locations.
        Run calculation procedure to generate results.
        """
        # Find zeros in matrix
        self._zero_locations = (matrix == 0)
        self._shape = matrix.shape

        # Choices starts without any choices made.
        self._choices = np.zeros(self._shape, dtype=bool)

        self._marked_rows = []
        self._marked_columns = []

        # marks rows and columns
        self.__calculate()

        # Draw lines through all unmarked rows and all marked columns.
        self._covered_rows = list(set(range(self._shape[0])) - set(self._marked_rows))
        self._covered_columns = self._marked_columns

    def get_covered_rows(self):
        """Return list of covered rows."""
        return self._covered_rows

    def get_covered_columns(self):
        """Return list of covered columns."""
        return self._covered_columns

    def __calculate(self):
        """
        Calculates minimum number of lines necessary to cover all zeros in a matrix.
        Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
        """
        while True:
            # Erase all marks.
            self._marked_rows = []
            self._marked_columns = []

            # Mark all rows in which no choice has been made.
            for index, row in enumerate(self._choices):
                if not row.any():
                    self._marked_rows.append(index)

            # If no marked rows then finish.
            if not self._marked_rows:
                return True

            # Mark all columns not already marked which have zeros in marked rows.
            num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

            # If no new marked columns then finish.
            if num_marked_columns == 0:
                return True

            # While there is some choice in every marked column.
            while self.__choice_in_all_marked_columns():
                # Some Choice in every marked column.

                # Mark all rows not already marked which have choices in marked columns.
                num_marked_rows = self.__mark_new_rows_with_choices_in_marked_columns()

                # If no new marks then Finish.
                if num_marked_rows == 0:
                    return True

                # Mark all columns not already marked which have zeros in marked rows.
                num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

                # If no new marked columns then finish.
                if num_marked_columns == 0:
                    return True

            # No choice in one or more marked columns.
            # Find a marked column that does not have a choice.
            choice_column_index = self.__find_marked_column_without_choice()

            while choice_column_index is not None:
                # Find a zero in the column indexed that does not have a row with a choice.
                choice_row_index = self.__find_row_without_choice(choice_column_index)

                # Check if an available row was found.
                new_choice_column_index = None
                if choice_row_index is None:
                    # Find a good row to accomodate swap. Find its column pair.
                    choice_row_index, new_choice_column_index = \
                        self.__find_best_choice_row_and_new_column(choice_column_index)

                    # Delete old choice.
                    self._choices[choice_row_index, new_choice_column_index] = False

                # Set zero to choice.
                self._choices[choice_row_index, choice_column_index] = True

                # Loop again if choice is added to a row with a choice already in it.
                choice_column_index = new_choice_column_index

    def __mark_new_columns_with_zeros_in_marked_rows(self):
        """Mark all columns not already marked which have zeros in marked rows."""
        num_marked_columns = 0
        for index, column in enumerate(self._zero_locations.T):
            if index not in self._marked_columns:
                if column.any():
                    row_indices, = np.where(column)
                    zeros_in_marked_rows = (set(self._marked_rows) & set(row_indices)) != set([])
                    if zeros_in_marked_rows:
                        self._marked_columns.append(index)
                        num_marked_columns += 1
        return num_marked_columns

    def __mark_new_rows_with_choices_in_marked_columns(self):
        """Mark all rows not already marked which have choices in marked columns."""
        num_marked_rows = 0
        for index, row in enumerate(self._choices):
            if index not in self._marked_rows:
                if row.any():
                    column_index, = np.where(row)
                    if column_index in self._marked_columns:
                        self._marked_rows.append(index)
                        num_marked_rows += 1
        return num_marked_rows

    def __choice_in_all_marked_columns(self):
        """Return Boolean True if there is a choice in all marked columns. Returns boolean False otherwise."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return False
        return True

    def __find_marked_column_without_choice(self):
        """Find a marked column that does not have a choice."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return column_index

        raise HungarianError(
            "Could not find a column without a choice. Failed to cover matrix zeros. Algorithm has failed.")

    def __find_row_without_choice(self, choice_column_index):
        """Find a row without a choice in it for the column indexed. If a row does not exist then return None."""
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            if not self._choices[row_index].any():
                return row_index

        # All rows have choices. Return None.
        return None

    def __find_best_choice_row_and_new_column(self, choice_column_index):
        """
        Find a row index to use for the choice so that the column that needs to be changed is optimal.
        Return a random row and column if unable to find an optimal selection.
        """
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            column_indices, = np.where(self._choices[row_index])
            column_index = column_indices[0]
            if self.__find_row_without_choice(column_index) is not None:
                return row_index, column_index

        # Cannot find optimal row and column. Return a random row and column.
        from random import shuffle

        shuffle(row_indices)
        column_index, = np.where(self._choices[row_indices[0]])
        return row_indices[0], column_index[0]