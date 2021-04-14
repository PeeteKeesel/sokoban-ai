import numpy as np

from typing import Tuple, List


# ================================================================
class MctsNode:
    """
    A Node of a Monte-Carlo Search Tree.
    Multiple connected MctsNodes form a Monte Carlo Tree.
    Each MctsNode represents an environment state, i.e. a state
    of the Sokoban grid world.
    """

    def __init__(self, roomState: Tuple[int], SokobanEnv: object,
                 parent: object, prevAction: int):
        """
        Initializes a Monte-Carlo tree search node which contains the current
        state of the board, the action which was taken to get to the current
        state, the node from which this action was taken and the children
        of the current state (the states after all feasible actions).

        Arguments:
            roomState  (Tuple[int]): state of the board.
            SokobanEnv (object):     defines the environment dynamics.
            parent     (object):     the parent MctsNode.
            prevAction (int):        the action which led to the current node.
        """

        self.roomState = roomState
        self.children = []
        self.parent = parent
        self.prevAction = prevAction

        self.SokobanEnv = SokobanEnv

    # ----------------------------------------------------------------
    # Get methods.

    def get_children(self):
        """
        Return a list of all children data of the current MctsNode.

        Returns:
            (list): list of the roomStates of the child MctsNodes.
        """
        return [child.data for child in self.children]

    def get_children_nodes(self):
        """
        Return a list of all Children Nodes of the current Node.

        Returns:
            (list): list of the Child Nodes.
        """
        return [child for child in self.children]

    # ----------------------------------------------------------------
    # Additional methods.

    def print_children(self):
        """ Prints all the Children for the MctsNode. """
        print(f"{self.roomState}: {self.get_children()}")

    def add_child(self, node: object):
        """
        Adds a child MctsNode to the current MctsNode.

        Arguments:
            node (object): child MctsNodes.
        """
        self.children.append(node)

    def add_children(self, nodes: List[object]):
        """
        Adds multiple children MctsNodes to the current MctsNode.

        Arguments:
            nodes (list): list of child nodes.
        """
        for node in nodes:
            self.children.append(node)


# ================================================================
class Mcts:
    """
    Class containing logic to execute Monte Carlo Tree Search from a given
    root src.algorithms.MctsNode.
    """

    def __init__(self, mctsNode):
        """
        Initializes a Monte-Carlo Tree Search object.

        Arguments:
             mctsNode (object): the src.algorithms.MctsNode to start the
                                Monte Carlo Tree Search from.
        """
        self.root = mctsNode


    def take_action(self, action: int):
        """
        Takes a specified action from the current root MctsNode such that
        the MctsNode after this action is the new root MctsNode.

        Arguments:
            action (int): action to take for the root MctsNode.
        """
        pass


    def selection(self):
        """
        Implements the Selection step of the MCTS.
        Applies UCB1 until some child nodes are non-existent (empty).

        """
        pass

    def simulation(self):
        pass

    def expansion(self):
        pass

    def backpropagation(self):
        pass

    def ucb1(self):
        pass


    def run_mcts(self, env: List, agentState: tuple) -> None:
        """
        - construct a tree for the given environment state @env if none exists
          yet. If one exists then use this one.

        Arguments:
            env        List  - The current state of the board.
            agentState tuple - The position of the agent on the board.
        """

        # ----------------------------------------------------------------
        # SELECTION.
        #   Traverse the tree from the root to a leaf balancing Exploitation
        #   and Exploration using UCT as the selection strategy.
        #   Exploitation: Choose move that leds to best results so far.
        #   Exploration:  Choose less promising moves.


        # ----------------------------------------------------------------
        # SIMULATION.
        #   Finish the game starting from the leaf node, playing
        #   psuedo-randomly based on heuristic knowledge.

        # ----------------------------------------------------------------
        # EXPANSION.
        #   Decide which nodes are stored in memory. e.g. expand one child per
        #   simulation. Expanded node = first encountered position taht was
        #   not present in the tree.

        # ----------------------------------------------------------------
        # BACKPROPAGATION.
        #   Propagate result of the simulation at the leaf node backwards to
        #   the root.

        ...

    def uct(self, some):
        """
        Selection strategy using Upper Confidence bounds applied to Tress
        (UTC) formula: bar{X} + C * sqrt{ln(t(N)) / t(N_i)}
        with bar{X} - average game value.
            N      - Node.
            t(N)   - number of times node N was visited.
            t(N_i) - number of times child N_i was visited.
        """
        pass