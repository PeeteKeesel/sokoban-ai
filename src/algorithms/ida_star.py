from utils import print_search_algorithm_results, HEURISTICS
from time import time
import sys


class HashTable:
    """
    Class to save already been visited room states to avoid unnecessary
    operations.
    """
    def __init__(self):
        self.table = {}

    def check_add(self, env):
        """
        Checks if the current room state has already visited. Returns a boolean
        value if it has been visited before or not. If not it adds the
        room state as a key to the hash-table.

        Args:
            env (MctsSokobanEnv): Sokoban environment holding the room state.

        Returns:
            True, if the room state was already added to the hash-table and
            thus been visited before. False, otherwise.
        """
        key = tuple(env.room_state.flatten())

        if key in self.table:
            return True
        else:
            self.table[key] = True
            return False

def is_closed(closedSet, x):
    """Checks if a given element {@x} is in a set {@closedSet}."""
    for y in closedSet:
        if x == y:
            return True
    return False


def ida_star_search(env, time_limit: int, heuristic: str, metrics: dict=None,
                    print_steps: bool=None):
    """
    Performs the iterative deepening A* algorithm. At each iteration, a
    depth-first search is performed, cutting off a branch when its total cost
    f(n) = g(n) + h(n) exceeds a given `threshold`. This threshold starts at
    the estimate of the cost at the initial state, and increases for each
    iteration of the algorithm. At each iteration, the threshld used for the
    next iteration is the minimum cost of all values that exceeded the current
    threshold.

    Args:
        env (MctsSokobanEnv): Extension of the SokobanEnv class holding the
                              dynamics off the environment.

        time_limit (int): The maximum time the algorithm is allowed to run.

        heuristic (str): Name of the heuristic method to use. Implemented
                         options can be found in
                         gym_sokoban/envs/mcts_sokoban_env.py

        metrics (dict): A dictionary containing relevant information about
                        the run of the algorithm.

        print_steps (bool): True, if partial steps should be printed, False,
                            otherwise.

    Returns:
        metrics, env: The updated metrics dictionary and the environment of the
                      final state. Does not have to be a terminal state since
                      the algorithm can stop e.g. after a specific time.
    """
    current_time = 0

    if not metrics:
        metrics = {
            'no_of_nodes_discovered': 0,  # The total number of discovered
                                          # nodes. Including repeated ones.
            'no_of_nodes_repeated': 0,  # The number of a times nodes got
                                        # discovered repeatedly.
            'nodes_explored': set(),  # The set of all discovered nodes
                                      # excluding duplications.
            'environemnts': set(),  # This saves the environment of the nodes.
            'action_traj': [],  # The trajectory of action taken.
            'time': 0  # The time it took until the current node.
        }

    if env.is_done():
        return metrics, env

    MAXNODES = 20_000_000
    openSet = []
    closedSet = []
    visitSet = []
    pathLimit = 0
    if heuristic == HEURISTICS["manhattan"]:
        pathLimit = env.manhattan_heuristic() - 1
    elif heuristic == HEURISTICS["hungarian"]:
        pathLimit = env.hungarian_heuristic()
    else:
        raise NotImplementedError(
            f"The specified heuristic `{heuristic}` is not implemented."
        )
    it = 0

    while True:
        start_time = time()

        pathLimit = pathLimit + 1
        print(f"current pathLimit: {pathLimit}")

        # Set the cost to travel from root to node n.
        env.g_value = 0

        openSet.insert(0, env)
        hash_table = HashTable()
        nodes = 0

        while len(openSet) > 0:
            start_time_inner = time()

            currentState = openSet.pop(0)
            metrics['action_traj'] = currentState.get_actions_lookup_chars(
                currentState.action_trajectory
            )
            print(f"currentState  {currentState.print_actions_as_chars(currentState.action_trajectory)}")
            currentState.render_colored()

            # Time limit reached.
            if current_time >= time_limit:
                print_search_algorithm_results("ida_star_search",
                                               currentState, metrics,
                                               "TIME LIMIT EXCEED")
                return metrics, None

            # Solution found.
            if currentState.all_boxes_on_target():
                print_search_algorithm_results("ida_star_search",
                                               currentState, metrics,
                                               "SOLUTION FOUND")
                return metrics, currentState

            nodes += 1
            metrics['no_of_nodes_discovered'] += 1
            if nodes % 1_000_000 == 0:
                print(f"{nodes/1_000_000} M nodes checked")
            if nodes == MAXNODES:
                print("Limit of nodes reached: exiting without a solution.")
                sys.exit(1)

            if heuristic == HEURISTICS["manhattan"]:
                currentState.f_value = currentState.g_value + currentState.manhattan_heuristic()
            else:
                currentState.f_value = currentState.g_value + currentState.hungarian_heuristic()
            # print(f"   manhattan: {currentState.manhattan_heuristic()}")
            # print(f"   hungarian: {currentState.hungarian_heuristic()}")
            # print(5*" "+f"   currentState.f_value = {currentState.f_value} ? {pathLimit} = pathLimit")
            if currentState.f_value <= pathLimit:
                closedSet.insert(0, currentState)
                print(10*" "+"children")
                # Update successor states' f and g values and add to open set
                # if it is not already in the closed set or has been generated.
                for child in currentState.get_children_environments():
                    child.render_colored()
                    # test if node has been "closed"
                    if is_closed(closedSet, child):
                        print(15*" "+"is closed")
                        continue

                    # Check if the node has already been generated.
                    if hash_table.check_add(child):
                        print(15 * " " + "has been checked")
                        metrics['no_of_nodes_repeated'] += 1
                        continue

                    child.g_value = currentState.g_value + 1
                    child.f_value = child.g_value + child.hungarian_heuristic()
                    openSet.insert(0, child)
            else:
                print(10*" "+"visitSet.insert()")
                visitSet.insert(0, currentState)

            # Update time.
            current_time += time() - start_time_inner
            metrics['time'] = current_time

        print(f"nodes checked: {nodes}")
        print(f"iteration: {it}")
        it = it + 1
        if len(visitSet) == 0:
            print("FAIL")
            return None, None

        # Set a new cut-off value (pathLimit).
        low = visitSet[0].f_value
        for x in visitSet:
            if x.f_value < low:
                low = x.f_value
        pathLimit = low

        # Move nodes from VisitSet to OpenSet and reset closedSet.
        openSet.extend(visitSet)
        visitSet = []
        closedSet = []

        # Update time and action trajectory.
        current_time += time() - start_time
        metrics['time'] = current_time
