"""
Basic structure is adapted from
https://github.com/tensorflow/minigo/blob/master/mcts.py
"""
import numpy as np
import collections
import math
import sys

from copy import deepcopy
from utils import SIMULATION_POLICIES


c_PUCT = 1.38   # Constant determining the level of exploration.
D_NOISE_ALPHA = 0.03  # Dirichlet noise alpha parameter to ensure exploration.
EPS_DIRICH = 0.25  # To handle when to add Dirichlet noise.
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly.
TEMP_THRESHOLD = 5

# Large constant used to ensure that rarely explored nodes are
# considered promising. Used for SP-UCT.
D = 1_000
# Constant used to balance exploration and exploitation.
C = np.sqrt(2)

# Probability for selection a random action. Used for the `eps-greedy`
# simulation policy.
EPS = 0.2

# Discount factor for the return.
GAMMA = 1 #0.9


class DummyNodeAboveRoot:
    """
    This class represents a special node that is used as the node above the
    initial root node.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.action_traj = []

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MctsNode:
    """
    This class represents a node in the Monte-Carlo search tree.
    """

    def __init__(self, Env, n_actions=np.array([1,2,3,4]), prev_action=None,
                 parent=None):
        self.Env = Env
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
        # print(5*" " + f"child_N = {self.child_N}")
        # print(5*" " + f"N = {self.N}")
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
    def sp_uct(self):
        """
        Single-player variant of the UCT algorithm. A child node is
        selected to maximize the outcome of this method. This formula is a
        modified UCT version from
        (https://dke.maastrichtuniversity.nl/m.winands/documents/KNOSYS_SameGame.pdf)
        """
        if np.all(self.child_N == 0):
            return np.repeat(np.PINF, self.n_actions)
        # print(f"  W = {self.W}  child_W = {self.child_W}")
        # print(f"                child_N = {self.child_N}")
        # print(f"                uct     = {self.uct}")
        return self.child_W / self.child_N\
               + C * np.sqrt(2 * np.log(self.N) /
                             self.child_N)\
               + np.sqrt((np.sum(self.child_N**2) -
                         self.child_N * (self.child_W / self.child_N)**2 + D) /
                         self.child_N)

    @property
    def sp_uct_second_term(self):
        return C * np.sqrt(2 * np.log(self.N) /
                             self.child_N)

    @property
    def uct(self):
        if np.all(self.child_N == 0):
            return np.repeat(np.PINF, self.n_actions)
        return self.child_W / self.child_N \
               + C * np.log(self.N) / self.child_N

    @property
    def N(self):
        """Returns the action which had been taken to get to this state."""
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

    ###########################################################################
    # Functionalities for the simple MCTS without using predictions of a      #
    # neural network.                                                         #
    ###########################################################################

    def select_and_expand(self):
        """
        Performs the Selection and Expansion step of SP-MCTS. In the Selection
        step the action which maximizes the SP-UCT formula is selected. If it
        was not already done before the node for that action will be added as a
        new child node to the current one.
        """
        current = self
        while not current.game_is_done():
            #current.N += 1

            # Leaf node encountered.
            if not current.is_expanded:
                #print(current.room_state)
                break

            feasible_actions = current.Env.get_non_deadlock_feasible_actions()
            #print(f"   select_and_expand() feasible_actions={feasible_actions}")

            # Current node is not fully expanded. Expand one random action.
            if len(current.children) < len(feasible_actions):
                return current.expand(feasible_actions)

            # Current node has no feasible actions to take.
            elif len(feasible_actions) == 0:
                #print("No feasible action from here!")
                return current, -self.Env.reward_finished - self.Env.reward_box_on_target

            # Current node is fully extended with its feasible actions.
            else:
                # Random tree policy if there are no values yet.
                if np.all(np.isposinf(current.sp_uct[feasible_actions])) or \
                    np.all(np.isnan(current.sp_uct[feasible_actions])):
                    rdm_action = np.random.choice(feasible_actions)
                    #print(f"rdm_action={rdm_action}")
                    current = current.maybe_add_child(rdm_action)
                else:
                    # Choose best child: the action with the highest score.
                    #current.Env.render_colored()
                    #print(f"current.sp_uct[{feasible_actions}]= {current.sp_uct[feasible_actions]}")
                    #print(5*" " + f"current.uct[{feasible_actions}]= {current.uct[feasible_actions]}")
                    #print(f"current.cAscor[{feasible_actions}]= {current.child_action_score[feasible_actions]}")
                    max_action_idx = np.nanargmax(
                        current.uct[feasible_actions] # TODO: change to sp_uct
                    )
                    max_action = feasible_actions[max_action_idx]
                    #current.Env.render_colored()
                    #print(5*" " + f"max_action={max_action} feasible_actions={feasible_actions}  max_action_idx={max_action_idx}  uct={current.uct}")
                    current = current.maybe_add_child(max_action)

        if current.game_is_done():
            return current, self.Env.reward_finished + self.Env.reward_box_on_target
        else:
            return current, 0


    def expand(self, actions):
        """
        Expands the current tree by adding a child node randomly from the set
        of child nodes which have not yet been visited.

        Args:
            actions (List[int]): A set of feasible actions.

        Returns:
            (MctsNode): The expanded MctsNode object.
        """
        untried_actions = set(actions) - set([child for child in self.children])
        random_action = np.random.choice(tuple(untried_actions))
        added_child = self.maybe_add_child(random_action)
        # print(f"  expand() untried_actions = {tuple(untried_actions)}\n"
        #       f"           added_child     = {random_action}\n"
        #       f"           reward_last     = {added_child.Env.reward_last}")
        return added_child, added_child.Env.reward_last

    def maybe_add_child(self, action):
        """
        Adds child node for {@action} if it does not exist yet, and returns it.
        Expand only nodes that represent a state that has not yet been visited.

        Returns:
            child node after taking {@action}
        """
        if action not in self.children:
            #print(f"action={action} not in self.children= {self.children}")
            new_env = deepcopy(self.Env)
            new_env.step(action)
            self.children[action] = MctsNode(
                new_env, new_env.get_n_actions(),
                prev_action=action, parent=self
            )

        return self.children[action]

    def revert_visits(self, up_to):
        """Undo the addition of visit counts."""
        self.N -= 1
        #print(f"self.revert_visits()  {self.N}")
        # if self.N < 0:
        #     #print(self.room_state)
        #     sys.exit(4)
        if self.parent is None or self is up_to: 
            return 
        self.parent.revert_visits(up_to)

    def perform_simulation(self, max_depth_sim, prev_reward, sim_policy="random"):
        """
        Performs the Simulation step of the SP-MCTS. Starting from the current
        MctsNode which should be a leaf node, we simulate until the game is
        done, thus, either the maximal number of steps is reached or the game
        is finished. The simulation is done using a specific simulation policy:
            - random: Selects a random action among those available in the
                      current state.
            - eps-greedy: Selects a random action with probability EPS or with
                          probability (1-EPS) the action that maximizes the
                          the reward of the resulting state with probability.

        Args:
            max_depth_sim (int): Maximal depth of the simulation tree.
                                 Describes how many steps a simulation should
                                 maximaly take.

            sim_policy (str): The simulation policy. Describes how actions will
                              be chosen during the simulation.

        Returns:
            (float): Total reward achieved by the simulation.
        """
        leaf = deepcopy(self)

        # Perform a rollout.
        tot_return_of_sim = prev_reward
        act_traj_sim, depth = [], 0
        done = leaf.game_is_done()
        while not done and depth < max_depth_sim:
            # Get a list of all feasible actions, excluding redundant and
            # deadlock actions.
            feasible_actions = leaf.Env.get_non_deadlock_feasible_actions()
            #print(10*" " + f"perform_simulation(): feasible_actions={feasible_actions}")
            if feasible_actions.size == 0:
                break

            sim_action = None
            if sim_policy == SIMULATION_POLICIES["random"]:
                sim_action = np.random.choice(feasible_actions)
                #print(13 * " " + f"perform_simulation(): taken={sim_action}")
            elif sim_policy == SIMULATION_POLICIES["eps-greedy"]:
                if np.random.uniform() < EPS:
                    sim_action = np.random.choice(feasible_actions)
                else:
                    sim_action = leaf.Env.get_best_immediate_action(feasible_actions)

            act_traj_sim.append(sim_action)

            # Make a random step in the environment.
            _, reward_last, done, _ = leaf.Env.step(sim_action)

            # Update total reward and return.
            tot_return_of_sim += reward_last #GAMMA**depth * reward_last
            #print(30*" " + f"reward_last(a={sim_action}): {round(reward_last, 4)}    tot_return_of_sim: {round(tot_return_of_sim, 4)}")
            #tot_reward_of_sim += reward_last
            #leaf.Env.update_total_return(depth, GAMMA)
            #assert tot_return_of_sim == leaf.Env.total_return

            depth += 1

        # if leaf.game_is_done():
        #     print("game is done")
        #     print(leaf.room_state)
        #     print(leaf.Env.room_state)
        # print(15*" " + f"simulation_trajectory= '{leaf.Env.print_actions_as_chars(act_traj_sim)}'\n" +\
        #       15*" " + f"received total_return= {tot_return_of_sim}\n" +\
        #       15*" " + f"heuristic manhattan= {leaf.Env.manhattan_heuristic()}\n" +\
        #       15*" " + f"heuristic hungarian= {leaf.Env.hungarian_heuristic()}\n" + \
        #       15 * " " + f"other heuristic  = {leaf.Env.other_heuristic()}\n" + \
        #       15*" " + f"             return= {round(tot_return_of_sim - 0.8, 4)}")

        return tot_return_of_sim + leaf.Env.other_heuristic() #leaf.Env.hungarian_heuristic()

    def backpropagate(self, value, up_to):
        """
        Performs the Backpropagation step of the SP-MCTS. The total reward
        obtained during the Simulation step is backpropagated through the
        tree, starting from the leaf node up to the root node {@up_to}.

        Args:
            value (float): The value to be backpropagated to the prev nodes.

            up_to (MctsNode): The node to which the value should be updated.
        """
        #print(f"                                backpropagate({value})")

        # if self.is_expanded:
        #     self.revert_visits(up_to=up_to)
            #return
        
        self.is_expanded = True
        #self.child_W = np.ones(self.n_actions, dtype=np.float32) * value
        self.child_W[0] = np.NINF
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a {@value} estimation to the node {@up_to}."""
        self.N += 1
        self.W += value
        #self.Env.render_colored()
        #print(f"  backup_value(value={value}) -> self.W = {self.W}")
        # Note: Q doesn't need to be set as in the paper.
        #       the property already handles the determination.
        if self.parent is None or self is up_to:
            return 
        self.parent.backup_value(value+self.Env.penalty_for_step, up_to)

    def game_is_done(self):
        # print("self.room_state")
        # print(self.room_state)
        # print("self.Env.get_room_state()")
        # print(self.Env.get_room_state())
        return self.Env.all_boxes_on_target()
        #return self.Env._check_if_done()

    # def select_until_leaf(self):
    #     current = self
    #     while True:
    #         current.N += 1
    #         # Leaf node is encountered. Because it has no children yet.
    #         if not current.is_expanded:
    #             break
    #         # Choose action with the highest upper confidence bound.
    #         max_action = np.argmax(current.child_action_score)
    #         # Add new child MctsNode if action was not taken before.
    #         current = current.maybe_add_child(max_action)
    #     return current
    # def add_virtual_loss(self, up_to):
    #     """Propagate a virtual loss {@up_to} a specific node."""
    #     self.n_vlosses += 1
    #     self.W -= 1
    #     if self.parent is None or self is up_to:
    #         return
    #     self.parent.add_virtual_loss(up_to)
    #
    # def revert_virtual_loss(self, up_to):
    #     """Undo the addition of virtual loss."""
    #     self.n_vlosses -= 1
    #     self.W += 1
    #     if self.parent is None or self is up_to:
    #         return
    #     self.parent.revert_virtual_loss(up_to)
    # # NEW
    # def perform_simulations(self, num_simulations):
    #     """
    #     Performs {@num_simulations} Simulation step of the SP-MCTS. Starting
    #     from the current MctsNode which should be a leaf node, we simulate
    #     until the game is done, thus, either the maximal number of steps is
    #     reached or the game is finished.
    #
    #     Arguments:
    #         num_simulations (int): The number of simulations to perform on each
    #                                leaf node.
    #         simulation_policy (str): The policy to execute the simulation.
    #     Returns:
    #
    #     """
    #     #print("perform_simulations() called!")
    #     reward_per_simulation = np.zeros(num_simulations)
    #
    #     for simulation in range(num_simulations):
    #         print(10*" " + f"for simulation in range(num_simulations) -> {simulation}  child_N={np.round(self.child_N,4)}   {len(self.children)}")
    #         leaf = deepcopy(self)
    #
    #         tot_reward_of_simulation, act_traj = 0, []
    #         # Perform the simulation.
    #         while True:
    #             # Get a list of all feasible actions, excluding redundant and
    #             # deadlock actions.
    #             non_deads = leaf.Env.get_non_deadlock_feasible_actions()  #TODO: can be removed
    #             random_action = np.random.choice(
    #                 leaf.Env.get_non_deadlock_feasible_actions()
    #             )
    #             act_traj.append(random_action)
    #                   #f"traj = {leaf_env.print_actions_as_chars(self.action_traj)}")
    #
    #             # Make a random step in the environment.
    #             _, reward_last, done, _ = leaf.Env.step(random_action)
    #
    #             print(15*" " + f"simulation {simulation} PICKED action {leaf.Env.print_actions_as_chars([random_action])} "
    #                   f" out of possible={non_deads}  "
    #                   f" after = {leaf.Env.print_actions_as_chars(act_traj)}  and received={reward_last}")
    #
    #             # Update the total reward.
    #             tot_reward_of_simulation += reward_last
    #
    #             # End the simulation if its done.
    #             if done:
    #                 print(15*" " + f"simulation {simulation} DONE after '{leaf.Env.print_actions_as_chars(leaf.action_traj + act_traj)}' tot_reward = {np.round(tot_reward_of_simulation,3)} of state \n{leaf.Env.room_state}")
    #                 break
    #
    #         # Update the total reward received for the previous simulation.
    #         reward_per_simulation[simulation] = tot_reward_of_simulation
    #
    #     # TODO: return the SUM, MEAN, ...?
    #     print(f"np.sum(reward_per_simulation) = {np.sum(reward_per_simulation)}\n\n")
    #     return np.sum(reward_per_simulation)

    # def perform_simulation(self, max_depth):
    #     """
    #     Performs the simulation step of the SP-MCTS. Starting from the current
    #     MctsNode which should be a leaf node, we simulate until the game is
    #     done, thus, either the maximal number of steps is reached or the game
    #     is finished.
    #
    #     Returns:
    #         (float): Total reward achieved by the simulation.
    #     """
    #     leaf = deepcopy(self)
    #
    #     # Perform a rollout.
    #     tot_reward_of_simulation, simulation_act_traj, depth = 0, [], 0
    #     while not leaf.game_is_done() and depth < max_depth:
    #         # Get a list of all feasible actions, excluding redundant and
    #         # deadlock actions.
    #         #non_deads = leaf.Env.get_non_deadlock_feasible_actions()  # TODO: can be removed
    #         random_action = np.random.choice(
    #             leaf.Env.get_non_deadlock_feasible_actions()
    #         )
    #         simulation_act_traj.append(random_action)
    #
    #         # Make a random step in the environment.
    #         _, reward_last, done, _ = leaf.Env.step(random_action)
    #         depth += 1
    #
    #         # Update the total reward.
    #         tot_reward_of_simulation += reward_last
    #
    #         # print(
    #         #     15 * " " + f"PICKED action {leaf.Env.print_actions_as_chars([random_action])} "
    #         #                f" out of possible={non_deads}  "
    #         #                f" after = {leaf.Env.print_actions_as_chars(simulation_act_traj)}  and received={reward_last}")
    #     print(15*" " + f"simulation_trajectory= '{leaf.Env.print_actions_as_chars(simulation_act_traj)}'\n" +\
    #           15*" " + f"received total_reward= {tot_reward_of_simulation}")
    #
    #     return tot_reward_of_simulation - leaf.Env.manhattan_heuristic()

    # def incorporate_nn_estimates(self, action_probs, value, up_to):
    #     """
    #     Incorporate the estimations of the neural network.
    #     This should be called if the node has just been expanded via
    #     `select_until_leaf`.
    #     """
    #     if self.is_expanded:
    #         self.revert_visits(up_to=up_to)
    #         return
    #     self.is_expanded = True
    #     self.original_P = self.child_P = action_probs
    #     self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
    #     self.backup_value(value, up_to=up_to)
    #
    # # NEW
    # def incorporate_action_probabilities(self, simulation_policy, up_to):
    #     if self.is_expanded:
    #         self.revert_visits(up_to=up_to)
    #         return
    #
    #     self.is_expanded = True
    #     # Selects a random action among those available in the current state
    #     if simulation_policy == "random":
    #         # The first action 'nothing' does not count into the probabilities
    #         self.original_P = self.child_P = self.get_random_probs()
    #         self.child_W = np.append([0],
    #                                  np.ones([self.n_actions-1], dtype=np.float32)\
    #                                  * self.Env.get_return()) # TODO: or is this reward_last
    #
    #     elif simulation_policy == "eps-greedy":
    #         raise NotImplementedError
    #     else:
    #         raise NotImplementedError
    #
    #     # self.child_W = np.arange(0, self.n_actions+1) # TODO

    # # NEW
    # def get_random_probs(self):
    #     """
    #     Returns probability distribution for the simulation policy 'random'.
    #     Thus, each action, except action 0 (doing nothing), has equal
    #     probability.
    #     """
    #     return np.append([0], np.repeat(1/(self.n_actions-1), self.n_actions-1))
    #
    # # NEW
    # def get_eps_greedy_probs(self):
    #     raise NotImplementedError
    #
    # # NEW
    # def incorporate_random_probs(self, action_probs, value, up_to):
    #     if self.is_expanded:
    #         self.revert_visits(up_to=up_to)
    #         return
    #     self.is_expanded = True
    #     self.original_P = self.child_P = action_probs
    #     # TODO: change reward_last to the value which in in the current state, so the total discounted reward
    #
    #     assert self.n_actions == len(self.children)
    #     self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
    #     # print(self.child_W)
    #     self.backup_value(value, up_to=up_to)

    # def select_until_leaf_random(self):
    #     """
    #     Selection step of the SP-MCTS. Starting from the current MctsNode this
    #     steps chooses the action which maximizes the SP-UCT formula until a
    #     leaf not is visited.
    #
    #     Returns:
    #         MctsNode - The encountered leaf node after the selection step.
    #     """
    #     #print(f"select_until_leaf_random() called!")
    #     current = self
    #     while True:
    #         # print(f"WHILE current.N={current.N}   current.Env.reward_last={current.Env.reward_last}")
    #         current.N += 1
    #         if not current.is_expanded:
    #             break
    #
    #         # this would choose the max action based on the probabilities
    #         # np.random.choice(np.flatnonzero(current.child_P == current.child_P.max()))
    #         # we just choose randomly here.
    #         # random_action = np.random.choice(np.arange(1, self.n_actions))
    #         # random_action = np.random.choice(current.Env.get_feasible_actions())
    #
    #         # Chooses the max action according to the SP-UCT score
    #         max_action = np.argmax(current.sp_uct)
    #         #print(f"     max_action={max_action}")
    #
    #         # Check for deadlocks and redundant action
    #         if current.Env.deadlock_detection(max_action) or \
    #             max_action not in current.Env.get_feasible_actions():
    #             continue
    #
    #         # # CHECK DEADLOCKS HERE
    #         # if current.Env.deadlock_detection(max_action):
    #         #     print(50*"#")
    #         #     print(len(current.children.keys()))
    #         #     print(f"Deadlock found for action {current.Env.get_action_lookup_chars(max_action)}")
    #         #     # Only add child if its not a deadlock, TODO: then change the random probabilities
    #         #     continue
    #         # if max_action not in current.Env.get_feasible_actions():
    #         #     continue
    #
    #         # Expansion step of the SP-MCTS: Expands the tree if not already.
    #         current = current.maybe_add_child(max_action)
    #     return current

    # def inject_noise(self):
    #     """
    #     Additional exploration is achieved by adding Dirichlet
    #     noise to the prior probabilities in the root node. This noise
    #     ensures that all actions may be trief, but the search may still
    #     overrule bad moves. (as in the paper)
    #     """
    #     dirich = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
    #     self.child_P = (1 - EPS_DIRICH) * self.child_P + EPS_DIRICH * dirich

    # def get_action_probs(self, squash=False, temperature=1):
    #     """
    #     Returns the child visit counts as a probability distribution.
    #     In the paper
    #         pi(a|s0) = N(s_0,a)^{1/temp} / sum(N(s_0,b)^{1/temp})
    #     where temp(erature) is a parameter that controls the level of
    #     exploration.
    #
    #     Arguments:
    #         squash (bool) - if True, exponentiate the probabilities
    #                         by a temperature slightly smaller than 1 to
    #                         encourage diversity in early steps.
    #         temperature (float) - for the first TODO: X (=30 in the paper)
    #                         moves it is 1. For the remainder of the game
    #                         an infinitesimely small value is used. We use 0.
    #      Returns:
    #         A policy vector containing probabilities for each of the n_actions.
    #     """
    #     probs = self.child_N
    #     if squash:
    #         probs = probs ** (1. / temperature) # .95
    #     return probs / np.sum(probs)

    def print_tree(self, depth=0):
        node = "|--- " + str(self.depth)
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
                f"\n      * score={np.around(self.child_action_score, 3)}" + \
                f"\n      * sp_uct={np.around(self.sp_uct, 3)}" + \
                f"\n      * uct={np.around(self.uct, 3)}"
        print(node)
        for _, child in sorted(self.children.items()):
            child.print_tree(depth+1)


class Mcts:
    """
    Represents a Monte-Carlo search tree and contains methods for
    performing the tree search.
    """

    def __init__(self, Env, num_parallel, simulation_policy, max_rollouts,
                 max_depth, agent_netw=None):
        """
        Arguments:
            Env (MctsSokobanEnv): Class that defines the environment dynamics.

            num_parallel (int): Number of leaf nodes to collect before
                                evaluating them in conjunction.

            simulation_policy (str): Name of the policy for performing a
                                     simulation.

            max_rollouts (int): Number of simulations per move or number of
                                traversals through the search tree before
                                performing a step, respectively.

            max_depth (int): Maximal depth of the simulation tree. This does
                             not include the depth of the search tree.

            agent_netw (...): Network for predicting action probabilities and
                              state value estimates.
        """
        self.Env = Env
        if agent_netw:
            self.agent_netw = agent_netw
        self.num_parallel = num_parallel
        self.temp_threshold = None
        self.simulation_policy = simulation_policy
        self.max_rollouts = max_rollouts
        self.max_depth = max_depth

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.train_examples = []

        self.hash_table = set()

        self.root = None

    def initialize_search(self):
        n_actions = self.Env.get_n_actions()
        self.root = MctsNode(self.Env, n_actions)

        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.train_examples = []

        self.hash_table = set()

    # def tree_search(self, num_parallel=None):
    #     """
    #     Performs multiple simulations in the tree (following trajectories)
    #     until a given amount of leaves to expand have been encountered.
    #     Then it expands and evalutes these leaf nodes.
    #     """
    #     if num_parallel is None:
    #         num_parallel = self.num_parallel
    #     leaves = []  # To save the leaf nodes which were expanded
    #     failsafe = 0
    #     while len(leaves) < num_parallel and failsafe < num_parallel * 2:
    #         failsafe += 1
    #         #self.root.print_tree()
    #         #print("_"*50)
    #         leaf = self.root.select_until_leaf()
    #         # If we encounter done-state, we do not need the agent network to
    #         # bootstrap. We can backup the value right away.
    #         if leaf.game_is_done():
    #             value = self.Env.get_return(leaf.Env.get_room_state(),
    #                                         leaf.depth)
    #             leaf.backup_value(value, up_to=self.root)
    #             continue
    #         # Otherwise, discourage other threads to take the same trajectory
    #         # via virtual loss and enqueue the leaf for evaluation by agent
    #         # network.
    #         leaf.add_virtual_loss(up_to=self.root)
    #         leaves.append(leaf)
    #
    #     # Evaluate the leaf-states all at once and backup the value estimates.
    #     if leaves:
    #         # 1st simulation policy: random
    #         if self.simulation_policy == "random":
    #             print(f"simulation_policy={self.simulation_policy}")
    #
    #         # 2nd simulation policy: epsilon-greedy
    #         elif self.simulation_policy == "eps-greedy":
    #             print(f"simulation_policy={self.simulation_policy}")
    #
    #         # 3rd simulation policy: neural network guided mcts
    #         elif self.simulation_policy == "alphago" and not self.agent_netw:
    #             print(f"simulation_policy={self.simulation_policy}")
    #             # TODO: implement neural network which predicts policy and value
    #             action_probs, values = self.agent_netw.step(
    #                 self.Env.get_obs_for_states([leaf.state for leaf in leaves]))
    #
    #             for leaf, action_prob, value in zip(leaves, action_probs, values):
    #                 leaf.revert_virtual_loss(up_to=self.root)
    #                 leaf.incorporate_nn_estimates(action_prob, value, up_to=self.root)
    #     return leaves
    #
    # def tree_search_random(self, num_parallel=None):
    #     """
    #     Performs multiple simulations in the tree (following trajectories)
    #     until a given amount of leaves to expand have been encountered.
    #     Then it expands and evalutes these leaf nodes.
    #     """
    #     #print("tree_search_random() called")
    #     if num_parallel is None:
    #         num_parallel = self.num_parallel
    #     leaves = []  # To save the leaf nodes which were expanded
    #     failsafe = 0
    #     while len(leaves) < num_parallel and failsafe < num_parallel * 2:
    #         failsafe += 1
    #
    #         # 1. Selection step + 2. Expansion step of the SP-MCTS. Starting
    #         # from root select until a leaf node is encountered using the
    #         # SP-UCT formula.
    #         leaf = self.root.select_and_expand()
    #
    #         # Break if the leaf node did not change.
    #         if np.alltrue(leaf.room_state == self.root.room_state):
    #             continue
    #
    #         # If we encounter done-state, we do not need the agent network to
    #         # bootstrap. We can backup the value right away.
    #         if leaf.game_is_done():
    #             print("is done")
    #             value = leaf.Env.get_return()
    #             #print(f"---total_reward={value}")
    #             leaf.backup_value(value, up_to=self.root)
    #
    #             # if self.root.parent.parent is not None:
    #             #     print(f"----------self.root.parent.child_W = {np.round(self.root.parent.child_W,3)}  {self.root.parent.prev_action}\n"
    #             #           f"          self.root.child_W        = {np.round(self.root.child_W, 3)}  {self.root.prev_action}")
    #             continue
    #
    #         # Otherwise, discourage other threads to take the same trajectory
    #         # via virtual loss and enqueue the leaf for evaluation by agent
    #         # network.
    #         #leaf.add_virtual_loss(up_to=self.root)
    #         leaves.append(leaf)
    #
    #     # Evaluate the leaf-states all at once and backup the value estimates.
    #     if leaves:
    #         # 1st simulation policy: random
    #         if self.simulation_policy == "random":
    #             for leafNode in leaves:
    #                 # 2. Simulation step of the SP-MCTS. Simulate using random
    #                 # actions until the game is done, which can be either by
    #                 # finishing the game or by reaching the max no. of steps.
    #                 tot_reward = leafNode.perform_simulations(self.max_rollouts)
    #
    #                 self.rollouts += 1 # TODO: nur eine simulation pro node durchfuehren?
    #                 print(f"rollouts={self.rollouts}")
    #
    #                 # Update the total value for the current leaf node.
    #                 leafNode.parent.child_W[leafNode.prev_action] = tot_reward
    #
    #                 leafNode.backpropagate(value=tot_reward, up_to=self.root)
    #         else:
    #             raise Exception("ERROR: FOR NOW WE ONLY TEST 'random' SIMULATION POLICY")
    #
    #         # # 2nd simulation policy: epsilon-greedy
    #         # elif self.simulation_policy == "eps-greedy":
    #         #     print(f"simulation_policy={self.simulation_policy}")
    #         #
    #         # # 3rd simulation policy: neural network guided mcts
    #         # elif self.simulation_policy == "alphago" and not self.agent_netw:
    #         #     print(f"simulation_policy={self.simulation_policy}")
    #         #     # TODO: implement neural network which predicts policy and value
    #         #     action_probs, values = self.agent_netw.step(
    #         #         self.Env.get_obs_for_states([leaf.state for leaf in leaves]))
    #         #
    #         #     for leaf, action_prob, value in zip(leaves, action_probs, values):
    #         #         leaf.revert_virtual_loss(up_to=self.root)
    #         #         leaf.incorporate_nn_estimates(action_prob, value, up_to=self.root)
    #     return leaves

    def monte_carlo_tree_search(self):
        """
        Performs a single Monte-Carlo-Tree-Search iteration. Thus, it selects
        from the current node until a leaf node using a selection policy. From
        that leaf it performs a simulation using a simulation policy and
        backpropagates the achieved value to the root node.
        """

        count, leaf_node, prev_reward = 0, None, 0
        while count < self.root.n_actions and leaf_node is None:

            # 1. Selection step + 2. Expansion step of the SP-MCTS. Starting
            # from root select until a leaf node is encountered using a
            # selection policy.
            leaf_node, prev_reward = self.root.select_and_expand()

            # Ignore non-changed room states and dont append them.
            # if np.alltrue(leaf_node.room_state == self.root.room_state):
            #     print("np.alltrue(leaf_node.room_state == self.root.room_state):")
            #     print(leaf_node.room_state)
            #     print(self.root.room_state)
            #     count += 1
            #     leaf_node = None
            #     continue

            # If we encounter done-state We can backup the value right away.
            # THIS SHOULD BE ALREADY INCLUDE IN select_and_expand()
            # if leaf_node.game_is_done():
            #     #print("game is done")
            #     value = leaf_node.parent.W + leaf_node.Env.reward_finished#leaf_node.W # leaf_node.Env.get_return()
            #     #print(f"    value={value}  traj={leaf_node.action_traj}")
            #     leaf_node.backpropagate(value, up_to=self.root)
            #     return
        #print(f"leaf_node after {self.Env.print_actions_as_chars(leaf_node.action_traj)}")
        #leaf_node.Env.render_colored()

        # Evaluate the child node and backup the value estimate.
        if leaf_node:
            # if leaf_node.depth == self.max_depth:
            #     print(f"MAX_DEPTH {leaf_node.depth} REACHED, reward={leaf_node.W} parent_reward={leaf_node.parent.W}")
            #     return

            # 3. Simulation step of the SP-MCTS. Simulate using random
            # actions until the game is done, which can be either by
            # finishing the game or by reaching the max no. of steps.
            if self.simulation_policy == SIMULATION_POLICIES["random"]:
                tot_return = leaf_node.perform_simulation(
                    max_depth_sim=self.max_depth,
                    prev_reward=prev_reward,
                    sim_policy=SIMULATION_POLICIES["random"]
                )
            elif self.simulation_policy == SIMULATION_POLICIES["eps-greedy"]:
                tot_return = leaf_node.perform_simulation(
                    max_depth_sim=self.max_depth,
                    prev_reward=prev_reward,
                    sim_policy=SIMULATION_POLICIES["eps-greedy"]
                )
            else:
                raise NotImplementedError("ERROR: Simulation policies other than 'random' not implemented yet.")

            # rs = tuple(leaf_node.room_state.flatten())
            # if rs in self.hash_table:
            #     print("         ALREADY visited")
            #     tot_return += leaf_node.Env.penalty_already_visited
            # else:
            #     self.hash_table.add(rs)
            #tot_return += prev_reward

            # Update the total value for the current leaf node.
            #leaf_node.parent.child_W[leaf_node.prev_action] = tot_return

            # 4. Backpropagatiom step of the SP-MCTS.
            #print(f"backpropagate total_return={tot_return}")
            leaf_node.backpropagate(value=tot_return, up_to=self.root)

            return
        raise Exception(f"ERROR: No leaf node found for {self.root.room_state}")


    # def pick_action(self):
    #     """
    #     Selects an action for the root state based on the visit counts.
    #     After a specific threshold only the actions with the highest visit
    #     count will be chosen. Before that threshold a random action can be
    #     chosen.
    #     """
    #     print(self.root.child_N)
    #     # NEW
    #     # action = np.argmax(self.root.child_W)
    #     #return np.argmax(self.root.child_W)
    #     if self.root.depth > 8: # TODO: insert max-depth paramater from where on only the best action will be chosen
    #         action = np.argmax(self.root.sp_uct)
    #         print(f"pick_action() returns {action}={self.Env.get_action_lookup_chars(action)}  because {self.root.sp_uct}")
    #     else:
    #         cdf = self.root.child_N.cumsum()
    #         cdf = cdf / cdf[-1]  # probabilities for each action depending on the
    #                              # visit counts.
    #         selection = random.random()
    #         action = cdf.searchsorted(selection)
    #         print(f"pick_action() returns {action}={self.Env.get_action_lookup_chars(action)}  from {self.root.child_N}")
    #         #print(50*"===")
    #         #self.root.print_tree()
    #         assert self.root.child_N[action] != 0
    #     return action
    #     # OLD - but this is correct TODO uncomment this
    #     # if self.root.depth > self.temp_threshold:
    #     #     print(f"if   {self.root.depth}")
    #     #     action = np.argmax(self.root.child_N)
    #     # else:
    #     #     print("else")
    #     #     cdf = self.root.child_N.cumsum()
    #     #     cdf = cdf / cdf[-1]  # probabilities for each action depending on the
    #     #                          # visit counts.
    #     #     selection = random.random()
    #     #     action = cdf.searchsorted(selection)
    #     #     print(f"action={action}   from {self.root.child_N}")
    #     #     #print(50*"===")
    #     #     #self.root.print_tree()
    #     #     assert self.root.child_N[action] != 0
    #     # print(f"pick_action() returns {action}={self.Env.get_action_lookup_chars(action)}  because {self.root.child_N}")
    #     # return action
    #
    # def take_action(self, action: int):
    #     """
    #     Takes a specified action from the current root MctsNode such that
    #     the MctsNode after this action is the new root MctsNode.
    #
    #     Arguments:
    #         action (int): action to take for the root MctsNode.
    #     """
    #     # Store data to be used as experience tuples.
    #     ob = self.Env.get_obs_for_states([self.root.room_state]) # TODO: implement get_obs_for_states
    #     #self.train_examples.append(ob)
    #     #self.searches_pi.append(self.root.get_action_probs()) # TODO: Use self.root.position.n < self.temp_threshold as argument
    #     self.qs.append(self.root.Q)
    #     # TODO: implement get_return function
    #     print(f"take_action(): {self.rewards}   {self.Env.get_return()}")
    #     reward = (self.Env.get_return(self.root.children[action].room_state,
    #                                   self.root.children[action].depth) - sum(self.rewards))
    #     self.rewards.append(reward)
    #
    #     # Resulting state becomes new root of the tree.
    #     self.root = self.root.maybe_add_child(action)
    #     del self.root.parent.children

    def take_best_action(self):
        """
        Performs multile Monte-Carlo-Tree-Search iterations after which it
        chooses and takes the best action and returns the corresponding
        information.

        Returns:
            observation (object): An env-specific object representing the
                                  observation of the environment.
            reward (float): The amount of reward achieved by the previous step.
            done (bool): Whether the game is finished or not.
            info (dict): Diagnostic information useful for debugging.
            best_action (int): The best action which was taken according to the
                               MCTS.
        """
        #print("take_best_action() called!")
        best_action = self.mcts()
        #print(f"best_action chosen is {best_action}")
        if best_action == -1:
            return None, -1, True, {"mcts_giveup": "MCTS Gave up, board unsolvable. No moves where found from here."}
        observation, reward, done, info = self.Env.step(best_action)
        return observation, reward, done, info, best_action


    def mcts(self):
        mcts_copy = deepcopy(self)

        rollouts = 0
        while rollouts < self.max_rollouts:
            if rollouts % 500 == 0:
                print(f"rollout = {rollouts}  root.child_N = {mcts_copy.root.child_N}")

            mcts_copy.monte_carlo_tree_search()
            rollouts += 1

        #temp = mcts_copy.root.sp_uct - mcts_copy.root.sp_uct_second_term
        # print(f"after {rollouts} rollouts we have child_N: {mcts_copy.root.child_N}\n" +
        #       21*" " + len(str(rollouts))*" " + f"and sp_uct: {mcts_copy.root.sp_uct}\n" +
        #       21*" " + len(str(rollouts))*" " + f"and    uct: {mcts_copy.root.uct}\n" +
        #       21*" " + len(str(rollouts))*" " + f"C=0 sp_uct: {temp}\n")
        # For the last action set C=0
        # best_child = np.nanargmax(mcts_copy.root.sp_uct -
        #                           mcts_copy.root.sp_uct_second_term)
        #best_child = np.nanargmax(mcts_copy.root.uct)
        best_child = np.nanargmax(mcts_copy.root.child_N)
        #mcts_copy.root.print_tree()
        print(f"Mcts choose best_child={best_child} because child_N = {mcts_copy.root.child_N}")
        return best_child

# def execute_episode_with_nnet(agentNetw, numSimulations, Env):
#     """
#     Executes a single episode of the task using Monte-Carlo tree search with
#     the given agent network. It returns the experience tuples collected during
#     the search.
#
#     Arguments:
#         agentNetw: Network for predicting action probabilities and state
#                        value estimates.
#         numSimulations: Number of simulations (traverses from root to leaf)
#                             per action.
#         Env: Environment that describes the environment dynamics.
#
#     Returns:
#     """
#     mcts = Mcts(agentNetw, Env)
#
#     mcts.initialize_search()
#
#     # Must run this once at the start, so that noise injection actually affects
#     # the first action of the episode.
#     firstNode = mcts.root.select_until_leaf()
#     probs, vals = agentNetw.step(
#         Env.get_obs_for_states([firstNode.state]))
#     firstNode.incorporate_nn_estimates(probs[0], vals[0], firstNode)
#
#     while True:
#         mcts.root.inject_noise()
#         currentSimulations = mcts.root.N  # the # of times the node was visited
#
#         # We want `num_simulations` simulations per action not counting
#         # simulations from previous actions.
#         while mcts.root.N < currentSimulations + numSimulations:
#             mcts.tree_search()
#
#         # mcts.root.print_tree()
#         # print("_"*100)
#
#         action = mcts.pick_action()
#         mcts.take_action(action)
#
#         if mcts.root.game_is_done():
#             break
#
#     # Computes the returns at each step from the list of rewards obtained at
#     # each step. The return is the sum of rewards obtained *after* the step.
#     # TODO: get_return needs to be implemented.
#     ret = [Env.get_return(mcts.root.state, mcts.root.depth) for _
#            in range(len(mcts.rewards))]
#
#     totalReward = np.sum(mcts.rewards)
#
#     obs = np.concatenate(mcts.obs)
#
#     return obs, mcts.searches_pi, ret, totalReward, mcts.root.state
#
#
# def execute_episode(numSimulations, Env, simulation_policy="random", max_rollouts=10, max_depth=20):
#
#     mcts = Mcts(Env, simulation_policy, max_rollouts, max_depth)
#     mcts.initialize_search()
#
#     # Must run this once at the start, so that noise injection actually affects
#     # the first action of the episode.
#     firstNode = mcts.root.select_and_expand()
#     firstNode.backpropagate(0, mcts.root)
#
#     #print(20*"#"+"\n" + mcts.root.print_tree() + "\n"+20*"#\n")
#     mcts.root.print_tree()
#
#     rollouts = 1
#     while rollouts <= mcts.max_rollouts:
#         print(f"\n\n--- Rollout {rollouts}")
#         # the # of times the node was visited
#         prevSimulations = mcts.root.N
#
#         # We want `num_simulations` simulations per action not counting
#         # simulations from previous actions.
#         while mcts.root.N < prevSimulations + numSimulations:
#             print(30*"**"+f" {mcts.root.N} < {prevSimulations} + {numSimulations} "+30*"**")
#             mcts.tree_search_random(num_simulations=numSimulations)
#
#         print(f" {mcts.root.N} > {prevSimulations} + {numSimulations} ")
#         print("_"*75+f" After {mcts.root.N-prevSimulations} simulations performed for the current node.")
#         mcts.root.print_tree()
#         print("_"*100)
#
#         action = mcts.pick_action()
#         print(f"    picked action {action}={Env.get_action_lookup_chars(action)} after action_traj={mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)}")
#         assert action != 0
#         mcts.take_action(action)
#         print(f"        reward={mcts.rewards}")
#
#         rollouts += 1
#
#         if mcts.root.Env._check_if_all_boxes_on_target():
#             print(f"After rollout {rollouts} and traj={mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)} ALL BOXES ARE ON TARGET!")
#             break
#
#         # if mcts.root.game_is_done():
#         #     print("++"*1000)
#         #     print(f"IF MCTS.ROOT.IS_DONE() after {prevSimulations} simulations with action_traj = {mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)}")
#         #     print("++" * 1000)
#         #     break
#
#     print(100*"_"+f"\n{rollouts} Rollouts performed.")
#     mcts.root.print_tree()



