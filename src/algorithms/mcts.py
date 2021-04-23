"""
Basic structure is adapted from
https://github.com/tensorflow/minigo/blob/master/mcts.py
"""
from copy import deepcopy
import random
import numpy as np
import collections
import math

from typing import Tuple, List

c_PUCT = 1.38   # Constant determining the level of exploration.
D_NOISE_ALPHA = 0.03  # Dirichlet noise alpha parameter to ensure exploration.
EPS = 0.25  # To handle when to add Dirichlet noise.
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly.
TEMP_THRESHOLD = 5

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
        self.child_N = np.zeros([n_actions], dtype=np.float32)
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
        return (c_PUCT * self.child_P * math.sqrt(1 + self.N) /
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
            new_Env = deepcopy(self.Env)
            new_Env.step(action)
            self.children[action] = MctsNode(
                new_Env, new_Env.get_n_actions(),
                prev_action=action, parent=self)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss {@up_to} a specific node."""
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """Undo the addition of virtual loss."""
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        self.N -= 1 
        if self.parent is None or self is up_to: 
            return 
        self.parent.revert_visits(up_to)

    def incorporate_nn_estimates(self, action_probs, value, up_to):
        """
        Incorporate the estimations of the neural network.
        This should be called if the node has just been expanded via
        `select_until_leaf`.
        """
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_P = self.child_P = action_probs
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a value estimation {@up_to} a node."""
        self.N += 1
        self.W += value
        # Note: Q doesn't need to be set as in the paper.
        #       the property already handles the determination.
        if self.parent is None or self is up_to:
            return 
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.Env._check_if_done()

    def inject_noise(self):
        """
        Additional exploration is achieved by adding Dirichlet
        noise to the prior probabilities in the root node. This noise
        ensures that all actions may be trief, but the search may still
        overrule bad moves. (as in the paper)
        """
        dirich = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_P = (1 - EPS) * self.child_P + EPS * dirich

    def child_visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.

        Arguments:
            squash (bool) - if True, exponentiate the probabilities
                            by a temperature slightly smaller than 1 to
                            encourage diversity in early steps.
        Returns:
            Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def print_tree(self, depth=0):
        node = "|--- " + str(depth)
        print(node)
        self.Env.render_colored()
        node =  f"Node: * prev_action={self.prev_action} = {self.Env.get_action_lookup_chars(self.prev_action)}" + \
                f"\n      * N={self.N}" + \
                f"\n      * W={self.W}" + \
                f"\n      * Q={self.child_Q}" + \
                f"\n      * P={self.child_P}" + \
                f"\n      * score={self.child_action_score}"
        print(node)
        for _, child in sorted(self.children.items()):
            child.print_tree(depth+1)

# ================================================================
class Mcts:
    """
    Represents a Monte-Carlo search tree and contains methods for
    performing the tree search.
    """

    def __init__(self, agent_netw, Env, simulations_per_move=300, num_parallel=8):
        """
        Arguments:
            agent_netw            (NN)         - Network for predicting action
                                                 probabilities and state value
                                                 estimates.
            Env                   (SokobanEnv) - Environment dynamics.
            simulations_per_move  (int)        - Number of traversals through the
                                                 tree before performing a step.
        """
        self.agent_netw = agent_netw
        self.Env = Env
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        n_actions = self.Env.get_n_actions()
        self.root = MctsNode(self.Env, n_actions)
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

    def tree_search(self, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        """
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []  # To save the leaf nodes which were expanded
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            # self.root.print_tree()
            # print("_"*50)
            leaf = self.root.select_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = self.Env.get_return(leaf.state, leaf.depth)
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
            # Evaluate the leaf-states all at once and backup the value estimates.
            if leaves:
                # TODO: implement neural network which predicts policy and value
                action_probs, values = self.agent_netw.step(
                    self.Env.get_obs_for_states([leaf.state for leaf in leaves]))
                for leaf, action_prob, value in zip(leaves, action_probs, values):
                    leaf.revert_virtual_loss(up_to=self.root)
                    leaf.incorporate_estimates(action_prob, value, up_to=self.root)
            return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        After a specific threshold only the actions with the highest visit
        count will be chosen. Before that threshold a random action can be
        chosen.
        """
        if self.root.depth > self.temp_threshold: 
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action

    def take_action(self, action: int):
        """
        Takes a specified action from the current root MctsNode such that
        the MctsNode after this action is the new root MctsNode.

        Arguments:
            action (int): action to take for the root MctsNode.
        """
        # Store data to be used as experience tuples.
        ob = self.Env.get_obs_for_states([self.root.state])
        self.obs.append(ob)
        self.searches_pi.append(
            self.root.visits_as_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        # TODO: imoplement get_return function
        reward = (self.Env.get_return(self.root.children[action].room_state, self.root.children[action].depth) - sum(self.rewards))
        self.rewards.append(reward)

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children

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

def execute_episode(agent_netw, num_simulations, Env):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
                       value estimates.
    :param num_simulations: Number of simulations (traverses from root to leaf)
                            per action.
    :param Env: Environment that describes the environment dynamics.
    :return:
    """
    mcts = Mcts(agent_netw, Env)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()
    probs, vals = agent_netw.step(
        Env.get_obs_for_states([first_node.state]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        # mcts.root.print_tree()
        # print("_"*100)

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    # TODO: get_return needs to be implemented.
    ret = [Env.get_return(mcts.root.state, mcts.root.depth) for _
           in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)

    return obs, mcts.searches_pi, ret, total_rew, mcts.root.state
