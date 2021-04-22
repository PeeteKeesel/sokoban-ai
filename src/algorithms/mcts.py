"""
Basic structure is adapted from
https://github.com/tensorflow/minigo/blob/master/mcts.py
"""
from copy import deepcopy

import numpy as np
import collections
import math

from typing import Tuple, List

c_puct = 1.38   # Constant determining the level of exploration.


class DummyNodeAboveRoot:

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


# ================================================================
class MctsNode:

    def __init__(self, Env, n_actions, prev_action=None, parent=None):
        self.Env = Env
        if parent is None:
            self.depth = 0
            parent = DummyNodeAboveRoot()
        else:
            self.depth = parent.depth + 1
        self.parent      = parent
        self.room_state  = self.Env.get_room_state()
        self.n_actions   = n_actions  # Number of actions from the node
        self.prev_action = prev_action  # The action which led to this node
        self.is_expanded = False  # If the node is already expanded or not
        self.n_vlosses   = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)  #
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_P = np.zeros([n_actions], dtype=np.float32)
        self.child_P    = np.zeros([n_actions], dtype=np.float32)
        self.children = {}

    @property
    def child_Q(self):
        """Returns the mean value of the state."""
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        """
        Returns U score of the state. The score is a variant of the
        PUCT algorithm from
        (http://gauss.ececs.uc.edu/Workshops/isaim2010/papers/rosin.pdf)
        """
        return (c_puct * self.child_P * math.sqrt(1 + self.N) /
                (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Returns the score of the state which will be used to select nodes
        in the selection step. The action for which this score is maximal
        will be chosen, thus, higher values are prefered in the search.
        As the upper confidence bound Q(s, a) + U(s, a) in the paper.
        """
        return self.child_Q + self.child_U

    @property
    def N(self):
        """Returns the action which led to this state had been taken."""
        return self.parent.child_N[self.prev_action]

    @property
    def W(self):
        """Returns the total action value for the state."""
        return self.parent.child_W[self.prev_action]

    @property
    def Q(self):
        """Returns the state action value Q."""
        return self.W / (1 + self.N)

    @N.setter
    def N(self, value):
        """Sets the number of times N the node has been visited."""
        self.parent.child_N[self.prev_action] = value

    @W.setter
    def W(self, value):
        """Sets the total action value W of the node."""
        self.parent.child_W[self.prev_action] = value

    def select_until_leaf(self):
        current = self
        while True:
            current.N += 1
            # Leaf node is encountered. Because it has no children yet.
            if not current.is_expanded:
                break
            # Choose action with the highest upper confidence bound.
            max_action = np.argmax(current.child_action_score)
            # Add new child MctsNode if action was not taken before.
            current    = current.maybe_add_child(max_action)
        return current

    def maybe_add_child(self, action):
        """
        Adds child node for {@action} if it does not exist yet, and returns it.
        """
        if action not in self.children:
            new_Env = deepcopy(self.Env).move(action)
            self.children[action] = MctsNode(
                new_Env, new_Env.get_n_actions(),
                prev_action=action, parent=self)
        return self.children[action]

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