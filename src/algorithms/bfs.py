from copy import deepcopy
from time import time
from src.utils import print_search_algorithm_results


def breadth_first_search(env, time_limit: int, metrics: dict=None, print_steps: bool=None):
    """
    Traverses the given {@env} in a breadth first search way until a termination condition is met.
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

    if env._check_if_done():
        return metrics, env

    env_queue = [env]  # this serves as the stack for the environments.

    while True:
        start_time = time()

        if not env_queue:
            print(metrics)
            raise Exception('depth_first_search(): Solution NOT FOUND! Empty environment queue.')

        node_env = env_queue.pop(0)
        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))

        if current_time >= time_limit:
            print_search_algorithm_results("breadth_first_search",
                                           node_env, metrics,
                                           "TIME LIMIT EXCEED")
            return metrics, None

        if node_env._check_if_all_boxes_on_target():
            print_search_algorithm_results("breadth_first_search",
                                           node_env, metrics,
                                           "SOLUTION FOUND")
            return metrics, node_env

        if node_env._check_if_maxsteps():
            print("Maximal number of steps reached!")
            continue

        children = node_env.get_children()

        if not [child for child in children if child is not None]:
            env_queue.pop(0)

        else:
            for action, child in enumerate(children):
                if print_steps:
                    if metrics["no_of_nodes_discovered"] % 1000 == 0:
                        print(f'no_of_nodes_discovered: {metrics["no_of_nodes_discovered"]}')
                metrics['no_of_nodes_discovered'] += 1

                child_env = deepcopy(node_env)  # copy the environment to take a "virtual" step
                child_env.step(action)          # take a step in the "virtual" environment

                if tuple(child_env.room_state.flatten()) not in metrics['nodes_explored'] and \
                   child_env                             not in env_queue:
                    env_queue.append(child_env)
                else:
                    metrics['no_of_nodes_repeated'] += 1

        # Update time and action trajectory.
        current_time += time() - start_time
        metrics['time'] = current_time
        metrics['action_traj'] = \
            node_env.get_actions_lookup_chars(node_env.action_trajectory)
