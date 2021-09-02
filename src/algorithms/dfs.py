from copy import deepcopy
from time import time
from src.utils import print_search_algorithm_results

import numpy as np


def depth_first_search(env, time_limit: int, metrics: dict=None, print_steps: bool=None):
    """
    Traverses the given {@env} in a depth first search way until a termination condition is met.

    Time: O(|V| + |E|)
    Space: O(|V|)
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
            #raise Exception('depth_first_search(): Solution NOT FOUND! Empty environment queue.')

        node_env = env_queue.pop(0)
        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))

        if current_time >= time_limit:
            print_search_algorithm_results("depth_first_search",
                                           node_env, metrics,
                                           "TIME LIMIT EXCEED")
            return metrics, None

        if node_env.all_boxes_on_target():
            print_search_algorithm_results("depth_first_search",
                                           node_env, metrics,
                                           "SOLUTION FOUND")
            return metrics, node_env

        if node_env.max_steps_reached():
            print("Maximal number of steps reached!")
            continue

        children = node_env.get_children()

        # every child is None.
        if not [child for child in children if child is not None]:
            env_queue.pop(0) # todo: Is this correct? Would pop a second time!

        # there is at least one feasible child which is not None.
        else:
            for action, child in reversed(list(enumerate(children))):
                if print_steps:
                    if metrics["no_of_nodes_discovered"] % 1000 == 0:
                        print(f'no_of_nodes_discovered: {metrics["no_of_nodes_discovered"]}')
                metrics['no_of_nodes_discovered'] += 1

                child_env = deepcopy(node_env)  # copy the environment to take a "virtual" step
                child_env.step(action)          # take a step in the "virtual" environment

                if tuple(child_env.room_state.flatten()) not in metrics['nodes_explored'] and \
                   not env_is_in_envs(child_env, env_queue):
                    env_queue.insert(0, child_env)
                else:
                    metrics['no_of_nodes_repeated'] += 1

        # Update time and action trajectory.
        current_time += time() - start_time
        metrics['time'] = current_time
        metrics['action_traj'] = \
            node_env.get_actions_lookup_chars(node_env.action_trajectory)


def env_is_in_envs(child_env, env_queue): 
    return np.any([np.alltrue(b.room_state == child_env.room_state) for b in env_queue])
