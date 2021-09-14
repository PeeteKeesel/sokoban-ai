
"""Monte-Carlo Tree Search implementation.

Raw means that at each leaf node a simulation is perfomed to evaluate the leaf,
as in the original MCTS.
"""
import numpy as np

from algorithms.mcts_node import MctsNode
from math import sqrt, log
from anytree import RenderTree


# Large constant used to ensure that rarely explored nodes are
# considered promising. Used for SP-UCT.
D = 1_000
# Constant used to balance exploration and exploitation. Used for UCT and
# SP-UCT.
C = sqrt(2)


class Mcts:
    def __init__(self, Env, actions, max_rollouts=4000, max_depth=30,
                 verbose=False):
        self.Env = Env
        self.step = 0
        self.max_rollouts = max_rollouts
        self.max_depth = max_depth
        self.actions = actions
        self.penalty_for_step = Env.penalty_for_step
        self.reward_finished = Env.reward_finished + Env.reward_box_on_target
        self.num_boxes = Env.num_boxes
        self.room_fixed = Env.room_fixed
        self.last_pos = Env.player_position
        self.moved_box = True
        self.verbose = verbose
        np.random.seed(1)


    def take_best_action(self, observation_mode="rgb_array"):
        """
        Runs multiple Monte-Carlo Tree Search iterations for the current state
        of the Sokoban environment.

        Returns:
            (observation, reward, done, info, best_action)
        """
        env_state = self.Env.get_current_state()
        best_action = self.mcts(env_state)

        # MCTS did not find any feasible move from the current position.
        if best_action == -1:
            return None, -1, True, {"mcts_giveup": "MCTS Gave up, board unsolvable. Reset board"}

        # Take the best action according the the MCTS.
        observation, reward, done, info = \
            self.Env.step(best_action, observation_mode=observation_mode)

        self.last_pos = env_state[2]
        self.moved_box = info["action.moved_box"]

        return observation, reward, done, info, best_action

    def mcts(self, env_state):
        """
        Performs a specified number of Monte-Carlo-Tree-Search iterations.
        Thus, it selects from the current node until a leaf node using a
        selection policy. From that leaf it performs a simulation using a
        simulation policy and backpropagates the achieved value to the root
        node.

        Args:
            env_state (np.arra): The room state of the Sokoban board.
        """
        root = MctsNode(name="0", state=env_state, last_pos=self.last_pos,
                    move_box=self.moved_box)
        rollouts = 0
        # Perform a specified number of rollouts.
        while rollouts <= self.max_rollouts:
            if rollouts % 500 == 0:
                print(f"rollout = {rollouts}   "
                      f"root.child_actions = {[child.action for child in root.children]}  "
                      f"root.child_N = {[child.N for child in root.children]}")

            # Selection and Expansion step of the Monte-Carlo Tree Search.
            child, immediate_reward = self.select_and_expand(root)

            # Board is unsolvable, if child is the root.
            if child.parent is None:
                return -1

            # Simulation step of the Monte-Carlo Tree Search.
            result = self.simulate(child, immediate_reward)

            # Backpropagation step of the Monte-Carlo Tree Search.
            self.back_propagate(result, child)

            rollouts += 1

        if self.verbose:
            for pre, fill, node in RenderTree(root):
                treestr = u"%s%s" % (pre, node.name)
                print(treestr.ljust(8), node.Q / node.N, node.N)

        # Find and return the action that got rolled out the most.
        best_child = max(root.children, key=lambda child: child.N)

        return best_child.action

    def select_and_expand(self, tree):
        """
        Performs the Selection and Expansion step of SP-MCTS. In the Selection
        step the action which maximizes the SP-UCT formula is selected. If it
        was not already done before the node for that action will be added as a
        new child node to the current one.
        """
        while not tree.done:
            sensible_actions = self.sensible_actions(player_pos=tree.state[2],
                                                     room_state=tree.state[3],
                                                     last_pos=tree.last_pos,
                                                     move_box=tree.move_box)

            if len([child.action for child in tree.children]) < len(sensible_actions):
                return self.expand(tree, sensible_actions)
            elif len(sensible_actions) == 0:
                return tree, -self.reward_finished
            else:
                tree = self.select(tree, C=C)

        if self.num_boxes == tree.state[0]:
            return tree, self.reward_finished
        else:
            return tree, 0

    def expand(self, node, sensible_actions):
        """
        Expands the current tree by adding a child node randomly from the set
        of child nodes which have not been visited yet and are feasible.

        Args:
              node (MctsNode): Object holding information about the current node.

              sensible_actions (List[int]): List of feasible action from the
                                            current node.

        Returns:
            new_child, reward_last:
                new_child (MctsNode): The MctsNode after taking an action.

                reward_last (float): The reward collected after taking the
                                     action.
        """
        untried_actions = set(sensible_actions) - \
                          set([child.action for child in node.children])
        action = np.random.choice(tuple(untried_actions))

        # Simulate a step taking the randomly chosen action.
        state, observation, reward_last, done, info = \
            self.Env.simulate_step(action=action, state=node.state)

        # Create a new MctsNode for the action taken.
        new_child = MctsNode(name=node.name + "-{}".format(action),
                             state=state, last_pos=node.state[2],
                             move_box=info["action.moved_box"], done=done,
                             parent=node, action=action)

        return new_child, reward_last

    def select(self, tree, selection_policy="uct", C=None):
        if selection_policy == "uct":
            return max(tree.children,
                       key=lambda child: self.uct(child, tree, C))
        elif selection_policy == "sp-uct":
            return max(tree.children,
                       key=lambda child: self.sp_uct(child, tree, C))
        else:
            return NotImplementedError


    @staticmethod
    def uct(child, tree, C_uct):
        return (child.Q / child.N) + (C_uct * log(tree.N) / child.N)

    @staticmethod
    def sp_uct(child, tree, C_sp_uct):
        return (child.Q / child.N) + (C_sp_uct * log(tree.N) / child.N)\
               + sqrt((sum([child.N**2 for child in tree.children])
                       - child.N*(child.Q / child.N)**2 + D)
                      / child.N)

    def simulate(self, node, immediate_reward):
        """
        Performs the Simulation step of the MCTS. Starting from the current
        MctsNode which should be a leaf node, it simulates until the game is
        done, thus, either the maximal number of steps is reached or the game
        is finished/solved. The simulation is done using a specific simulation
        policy:
            - random: Selects a random action among those available in the
                      current state.
            - eps-greedy: Selects a random action with probability EPS or with
                          probability (1-EPS) the action that maximizes the
                          the reward of the resulting state with probability.

        Args:
            node (MctsNode): The leaf node from where to perform the simulation.

            immediate_reward (float): The reward to be in the current state.

        Returns:
            (float): Total reward collected by the simulation plus a heuristic
                     estimate of the last state of the simulation.
        """

        depth = 0
        total_reward = immediate_reward
        state = node.state
        done = node.done
        last_pos = node.last_pos
        move_box = node.move_box

        # Start simulation.
        while not done and depth < self.max_depth:

            # Get all feasible action from the current state.
            possible_actions = self.sensible_actions(state[2], state[3], last_pos, move_box)
            #list(self.Env.get_non_deadlock_feasible_actions())#self.sensible_actions(state[2], state[3], last_pos, move_box)
            #sensible_actions = self.sensible_actions(state[2], state[3], last_pos, move_box)

            # print(possible_actions)
            #print(type(sensible_actions))
            # No feasible action from the current state.
            if not possible_actions:
                break

            action = np.random.choice(possible_actions)
            new_state, observation, reward, done, info = self.Env.simulate_step(action=action, state=state)
            last_pos = state[2]
            move_box = info["action.moved_box"]
            state = new_state
            total_reward += reward
            depth += 1

        return total_reward + self.heuristic(state[3])

    def heuristic(self, room_state):
        """
        The sum of the Manhattan distances of each box to its nearest goal.

        Args:
            room_state (np.array): Room state of the Sokoban board.

        Returns:
            (float): Sum of the Manhattan distance of each to to its nearest.
                     goal descaled by the penalty of a step.
        """
        total = 0
        arr_goals = (self.room_fixed == 2)
        arr_boxes = ((room_state == 4) + (room_state == 3))
        # find distance between each box and its nearest storage
        for i in range(len(arr_boxes)):
            for j in range(len(arr_boxes[i])):
                if arr_boxes[i][j] == 1:  # found a box
                    min_dist = 9999999
                    # check every storage
                    for k in range(len(arr_goals)):
                        for l in range(len(arr_goals[k])):
                            if arr_goals[k][l] == 1:  # found a storage
                                min_dist = min(min_dist, abs(i - k) + abs(j - l))
                    total = total + min_dist
        return total * self.penalty_for_step

    def back_propagate(self, value, node):
        """
        Performs the Backpropagation step of the MCTS. The total reward
        {@result} obtained during the Simulation step is backpropagated
        through the tree, starting from the leaf node {@node} up to the
        root node.

        Args:
            value (float): The value to be backpropagated to the prev nodes.

            node (MctsNode): The current node from which to backpropagate until
                         the root node.
        """
        while node is not None:
            node.Q += value
            node.N += 1
            value += self.penalty_for_step
            node = node.parent

    def sensible_actions(self, player_pos, room_state, last_pos, move_box):
        def sensible(action, room_state, player_position, last_pos, move_box):
            change = CHANGE_COORDINATES[action - 1]
            new_pos = player_position + change
            # Next position is a wall.
            if room_state[new_pos[0], new_pos[1]] == 0:
                return False
            if np.array_equal(new_pos, last_pos) and not move_box:
                return False
            new_box_position = new_pos + change
            # Box is already at a wall.
            if new_box_position[0] >= room_state.shape[0] \
                    or new_box_position[1] >= room_state.shape[1]:
                return False
            can_push_box = room_state[new_pos[0], new_pos[1]] in [3, 4]
            can_push_box &= room_state[new_box_position[0], new_box_position[1]] in [1, 2]
            if can_push_box:
                # Pushing a box into a corner.
                if self.room_fixed[new_box_position[0], new_box_position[1]] != 2:
                    box_surroundings_walls = []
                    for i in range(4):
                        surrounding_block = new_box_position + CHANGE_COORDINATES[i]
                        if self.room_fixed[surrounding_block[0], surrounding_block[1]] == 0:
                            box_surroundings_walls.append(True)
                        else:
                            box_surroundings_walls.append(False)
                    if box_surroundings_walls.count(True) >= 2:
                        if box_surroundings_walls.count(True) > 2:
                            return False
                        if not ((box_surroundings_walls[0] and box_surroundings_walls[1]) or (
                                box_surroundings_walls[2] and box_surroundings_walls[3])):
                            return False
            # Trying to push box into wall.
            if room_state[new_pos[0], new_pos[1]] in [3, 4] and room_state[
                new_box_position[0], new_box_position[1]] not in [1, 2]:
                return False
            return True

        return [action for action in self.actions if sensible(action, room_state, player_pos, last_pos, move_box)]


CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
