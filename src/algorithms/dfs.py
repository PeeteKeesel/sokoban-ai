from copy import deepcopy
from time import time
from src.utils import get_search_algorithm_results, \
    MESSAGE_SOLUTION_FOUND, MESSAGE_TIME_LIMIT_REACHED, ALGORITHM_NAME_DFS, \
    METRICS_SCELETON

import numpy as np


def depth_first_search(env, time_limit: int, metrics: dict=None, print_steps: bool=False):
    """
    Traverses the given {@env} in a depth first search way until a termination condition is met.

    Time: O(|V| + |E|)
    Space: O(|V|)

    Args:
        env (MctsSokobanEnv): Extension of the SokobanEnv class holding the
                              dynamics off the environment.

        time_limit (int): The maximum time the algorithm is allowed to run.

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
        metrics = METRICS_SCELETON

    if env.is_done():
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
            return get_search_algorithm_results(ALGORITHM_NAME_DFS,
                                                node_env, metrics,
                                                MESSAGE_TIME_LIMIT_REACHED)

        if node_env.all_boxes_on_target():
            return get_search_algorithm_results(ALGORITHM_NAME_DFS,
                                                node_env, metrics,
                                                MESSAGE_SOLUTION_FOUND)

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
