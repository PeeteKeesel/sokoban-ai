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

from gym_sokoban.envs import SokobanEnv

c_PUCT = 1.38   # Constant determining the level of exploration.
D_NOISE_ALPHA = 0.03  # Dirichlet noise alpha parameter to ensure exploration.
EPS = 0.25  # To handle when to add Dirichlet noise.
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly.
TEMP_THRESHOLD = 5

# large constant used to ensure that rarely explored nodes are
# considered promising. Used for SP-UCT.
D = 10 # TODO: what value to choose?
C = 2

class DummyNodeAboveRoot:

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.action_traj = []

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


# ================================================================
class MctsNode:

    def __init__(self, Env, n_actions, prev_action=None, parent=None):
        self.Env = deepcopy(Env)
        if parent is None:
            self.depth = 0
            parent = DummyNodeAboveRoot()
            self.action_traj = []
        else:
            self.depth = parent.depth + 1
            if prev_action:
                self.action_traj = parent.action_traj + [prev_action]
            else:
                self.action_traj = []
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

    # NEW
    @property
    def sp_uct(self):
        """
        Single-player variant of the UCT algorithm. A child node is
        selected to maximize the outcome of this method.
        """
        return self.child_Q / self.child_N\
               + C * np.sqrt(2 * np.log(self.N) /
                             self.child_N)\
               + np.sqrt((np.sum(self.child_N**2) -
                         self.child_N * (self.child_Q / self.child_N)**2 + D) /
                         self.child_N)

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

    # NEW
    def select_until_leaf_random(self):
        current = self
        while True:
            print(f"WHILE current.N={current.N}   current.Env.reward_last={current.Env.reward_last}")
            current.N += 1
            if not current.is_expanded:
                break

            # this would choose the max action based on the probabilities
            # np.random.choice(np.flatnonzero(current.child_P == current.child_P.max()))
            # we just choose randomly here.
            random_action = np.random.choice(np.arange(1, self.n_actions))
            # TODO: only action which are feasible, thus NO DEADLOCKS nor non-changing-actions
            # CHECK DEADLOCKS HERE
            if current.Env.deadlock_detection(random_action):
                print(50*"#")
                print(len(current.children.keys()))
                print(f"Deadlock found for action {current.Env.get_action_lookup_chars(random_action)}")
                # Only add child if its not a deadlock, TODO: then change the random probabilities
                continue
            # CHECK IF ACTION IS FEASIBLE
            current = current.maybe_add_child(random_action)
        return current

    # NEW
    def select_until_leaf_eps_greedy(self):
        raise NotImplementedError

    def maybe_add_child(self, action):
        """
        Adds child node for {@action} if it does not exist yet, and returns it.

        Returns:
            child node after taking {@action}
        """
        if action not in self.children:
            new_Env = deepcopy(self.Env)
            print(f"self.Env.reward_last={self.Env.reward_last}")
            new_Env.step(action)
            print(f"new_Env.reward_last={new_Env.reward_last}")
            self.children[action] = MctsNode(
                new_Env, new_Env.get_n_actions(),
                prev_action=action, parent=self)

        print(f"----------- {self.children[action].Env.reward_last}")
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
        """Undo the addition of visit counts."""
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

    # NEW
    def incorporate_action_probabilities(self, simulation_policy, up_to):
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return

        self.is_expanded = True
        # Selects a random action among those available in the current state
        if simulation_policy == "random":
            # The first action 'nothing' does not count into the probabilities
            self.original_P = self.child_P = self.get_random_probs()
            self.child_W = np.append([0],
                                     np.ones([self.n_actions-1], dtype=np.float32)\
                                     * self.Env.get_return()) # TODO: or is this reward_last

        elif simulation_policy == "eps-greedy":
            raise NotImplementedError
        else:
            raise NotImplementedError

        # self.child_W = np.arange(0, self.n_actions+1) # TODO


    # NEW
    def get_random_probs(self):
        """
        Returns probability distribution for the simulation policy 'random'.
        Thus, each action, except action 0 (doing nothing), has equal
        probability.
        """
        return np.append([0], np.repeat(1/(self.n_actions-1), self.n_actions-1))

    # NEW
    def get_eps_greedy_probs(self):
        raise NotImplementedError

    # NEW
    def incorporate_random_probs(self, action_probs, value, up_to):
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_P = self.child_P = action_probs
        # TODO: change reward_last to the value which in in the current state, so the total discounted reward

        assert self.n_actions == len(self.children)
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        print(self.child_W)
        self.backup_value(value, up_to=up_to)
        

    def backup_value(self, value, up_to):
        """Propagates a value estimation to the node {@up_to}."""
        self.N += 1
        self.W += value
        # Note: Q doesn't need to be set as in the paper.
        #       the property already handles the determination.
        if self.parent is None or self is up_to:
            return 
        self.parent.backup_value(value, up_to)

    def is_done(self):
        if self.Env._check_if_done():
            print(100*"+")
            print("IS_DONE")
            print(self.Env.render_colored())
            print(100 * "-")
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

    def get_action_probs(self, squash=False, temperature=1):
        """
        Returns the child visit counts as a probability distribution.
        In the paper
            pi(a|s0) = N(s_0,a)^{1/temp} / sum(N(s_0,b)^{1/temp})
        where temp(erature) is a parameter that controls the level of
        exploration.

        Arguments:
            squash (bool) - if True, exponentiate the probabilities
                            by a temperature slightly smaller than 1 to
                            encourage diversity in early steps.
            temperature (float) - for the first TODO: X (=30 in the paper)
                            moves it is 1. For the remainder of the game
                            an infinitesimely small value is used. We use 0.
         Returns:
            A policy vector containing probabilities for each of the n_actions.
        """
        probs = self.child_N
        if squash:
            probs = probs ** (1. / temperature) # .95
        return probs / np.sum(probs)

    def print_tree(self, depth=0):
        node = "|--- " + str(depth)
        print(node)
        self.Env.render_colored()
        node =  f"Node: * prev_action={self.prev_action} = {self.Env.get_action_lookup_chars(self.prev_action)}" + \
                f"\n      * action_traj= {self.Env.print_actions_as_chars(self.action_traj)}" + \
                f"\n      * N={self.N}" + \
                f"\n      * W={self.W}" + \
                f"\n      * Q={self.Q}" + \
                f"\n      * child_N={self.child_N}" + \
                f"\n      * child_W={np.around(self.child_W, 3)}" + \
                f"\n      * child_Q={np.around(self.child_Q, 3)}" + \
                f"\n      * child_P={np.around(self.child_P, 3)}" + \
                f"\n      * score={np.around(self.child_action_score, 3)}"
        print(node)
        for _, child in sorted(self.children.items()):
            child.print_tree(depth+1)

# ================================================================
class Mcts:
    """
    Represents a Monte-Carlo search tree and contains methods for
    performing the tree search.
    """

    def __init__(self, Env, agent_netw=None, simulations_per_move=300, num_parallel=8, simulation_policy="random"):
        """
        Arguments:
            agent_netw            (NN)         - Network for predicting action
                                                 probabilities and state value
                                                 estimates.
            Env                   (SokobanEnv) - Environment dynamics.
            simulations_per_move  (int)        - Number of traversals through the
                                                 tree before performing a step.
        """
        self.Env = Env
        if agent_netw:
            self.agent_netw = agent_netw
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None
        self.simulation_policy = simulation_policy

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.train_examples = []

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
        self.train_examples = []

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
            #self.root.print_tree()
            #print("_"*50)
            leaf = self.root.select_until_leaf()
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                value = self.Env.get_return(leaf.Env.get_room_state(),
                                            leaf.depth)
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)

        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            # 1st simulation policy: random
            if self.simulation_policy == "random":
                print(f"simulation_policy={self.simulation_policy}")

            # 2nd simulation policy: epsilon-greedy
            elif self.simulation_policy == "eps-greedy":
                print(f"simulation_policy={self.simulation_policy}")

            # 3rd simulation policy: neural network guided mcts
            elif self.simulation_policy == "alphago" and not self.agent_netw:
                print(f"simulation_policy={self.simulation_policy}")
                # TODO: implement neural network which predicts policy and value
                action_probs, values = self.agent_netw.step(
                    self.Env.get_obs_for_states([leaf.state for leaf in leaves]))

                for leaf, action_prob, value in zip(leaves, action_probs, values):
                    leaf.revert_virtual_loss(up_to=self.root)
                    leaf.incorporate_nn_estimates(action_prob, value, up_to=self.root)
        return leaves

    def tree_search_random(self, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        """
        print("tree_search_random() called")
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []  # To save the leaf nodes which were expanded
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            #self.root.print_tree()
            #print("_"*50)
            leaf = self.root.select_until_leaf_random()
            print(f"   tree_search_random(): LEAF selected.  {self.root.depth}  {self.root.action_traj}")
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.
            if leaf.is_done():
                print(f"//// if leaf.is_done()    \n {self.Env.render_colored()}  {self.root.action_traj}")
                value = self.Env.get_return(leaf.Env.get_room_state(),
                                            leaf.depth)
                print(f"---total_reward={value}")
                leaf.backup_value(value, up_to=self.root)
                continue
            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            #leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)

        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            # 1st simulation policy: random
            if self.simulation_policy == "random":
                print(f"simulation_policy={self.simulation_policy}")
                for leave in leaves:
                    leave.incorporate_action_probabilities("random", self.root)
            else:
                raise Exception("ERROR: FOR NOW WE ONLY TEST 'random' SIMULATION POLICY")

            # # 2nd simulation policy: epsilon-greedy
            # elif self.simulation_policy == "eps-greedy":
            #     print(f"simulation_policy={self.simulation_policy}")
            #
            # # 3rd simulation policy: neural network guided mcts
            # elif self.simulation_policy == "alphago" and not self.agent_netw:
            #     print(f"simulation_policy={self.simulation_policy}")
            #     # TODO: implement neural network which predicts policy and value
            #     action_probs, values = self.agent_netw.step(
            #         self.Env.get_obs_for_states([leaf.state for leaf in leaves]))
            #
            #     for leaf, action_prob, value in zip(leaves, action_probs, values):
            #         leaf.revert_virtual_loss(up_to=self.root)
            #         leaf.incorporate_nn_estimates(action_prob, value, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        After a specific threshold only the actions with the highest visit
        count will be chosen. Before that threshold a random action can be
        chosen.
        """
        print(self.root.child_N)
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf = cdf / cdf[-1]  # probabilities for each action depending on the
                                 # visit counts.
            selection = random.random()
            action = cdf.searchsorted(selection)
            #print(50*"===")
            #self.root.print_tree()
            assert self.root.child_N[action] != 0
        print(f"pick_action() returns {action}={self.Env.get_action_lookup_chars(action)}")
        return action

    def take_action(self, action: int):
        """
        Takes a specified action from the current root MctsNode such that
        the MctsNode after this action is the new root MctsNode.

        Arguments:
            action (int): action to take for the root MctsNode.
        """
        # Store data to be used as experience tuples.
        ob = self.Env.get_obs_for_states([self.root.room_state]) # TODO: implement get_obs_for_states
        self.train_examples.append(ob)
        self.searches_pi.append(
            self.root.get_action_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        # TODO: imoplement get_return function
        reward = (self.Env.get_return(self.root.children[action].room_state,
                                      self.root.children[action].depth) - sum(self.rewards))
        self.rewards.append(reward)

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children


def execute_episode_with_nnet(agentNetw, numSimulations, Env):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.

    Arguments:
        agentNetw: Network for predicting action probabilities and state
                       value estimates.
        numSimulations: Number of simulations (traverses from root to leaf)
                            per action.
        Env: Environment that describes the environment dynamics.

    Returns:
    """
    mcts = Mcts(agentNetw, Env)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    firstNode = mcts.root.select_until_leaf()
    probs, vals = agentNetw.step(
        Env.get_obs_for_states([firstNode.state]))
    firstNode.incorporate_nn_estimates(probs[0], vals[0], firstNode)

    while True:
        mcts.root.inject_noise()
        currentSimulations = mcts.root.N  # the # of times the node was visited

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < currentSimulations + numSimulations:
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

    totalReward = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)

    return obs, mcts.searches_pi, ret, totalReward, mcts.root.state


def execute_episode(numSimulations, Env, simulation_policy="random"):

    mcts = Mcts(Env, simulation_policy)
    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    firstNode = mcts.root.select_until_leaf_random()
    firstNode.incorporate_action_probabilities(simulation_policy, firstNode)

    # firstNode.incorporate_random_probs(np.repeat(1/mcts.root.n_actions,
    #                                              mcts.root.n_actions),
    #                                    mcts.Env.reward_last, firstNode)
    print("################# START #################")
    print(mcts.root.print_tree())
    print("################# END ###################")

    while True:
        #mcts.root.inject_noise()
        currentSimulations = mcts.root.N  # the # of times the node was visited
        print(f"---currentSimulation={currentSimulations}")

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < currentSimulations + numSimulations:
            mcts.tree_search_random()

        mcts.root.print_tree()
        print("_"*100)

        action = mcts.pick_action()
        print(f"... picked action {action}={Env.get_action_lookup_chars(action)}")
        print(mcts.rewards)
        mcts.take_action(action)

        if mcts.root.is_done():
            break


    print("after")

    pass