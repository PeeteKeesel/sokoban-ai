from copy             import deepcopy
from typing           import List
from gym_sokoban.envs import SokobanEnv

import numpy as np


CURRENT_BEST = []
steps_in_depth = 0

def depth_first_search_recursive(env: SokobanEnv, metrics: dict() = None, print_steps: bool = None):
    """
    Traverses the given {@env} in a depth first search way until a termination condition is met.

    Time: O(|V| + |E|)
    Space: O(|V|)
    """
    global CURRENT_BEST, max_steps_in_depth, steps_in_depth
    max_steps_in_depth = 50

    if not metrics:
        metrics = {
            'no_of_nodes_discovered': 0,  # the total number of discovered nodes. Including repeated ones.
            'no_of_nodes_repeated': 0,  # the number of a times nodes got discovered repeatedly.
            'nodes_explored': set(),  # the set of all discovered nodes excluding duplications.
            'environemnts': set(),  # this saves the environment of the nodes.
        }

    if env._check_if_done():
        print(f"{env.action_trajectory}")
        return env.action_trajectory
        #return env.action_trajectory, metrics, env

    env_queue = [env]  # this serves as the stack for the environments.

    while True:
        print(f"CURRENT_BEST = {CURRENT_BEST}")

        if not env_queue:
            return []
            #print(metrics)
            #return [], False, False
            # raise Exception('depth_first_search(): Solution NOT FOUND! Empty environment queue.')

        node_env = env_queue.pop(0)
        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))
        flag = False
        if tuple(node_env.room_state.flatten()) == tuple(TEST.flatten()):
            flag = True
            print("flag -> True")
        #print(f"WHILE: len(env_queue)={len(env_queue)}  \n{node_env.room_state}\n")

        if node_env._check_if_all_boxes_on_target():
            print("------------------------------------------------------------------------------------\n" +
                  f"depth_first_search_recursive(): Solution FOUND! Got {len(node_env.action_trajectory)} steps\n" +
                  f"{node_env.action_trajectory}\n" +
                  f"discovered: {metrics['no_of_nodes_discovered']}\n" +
                  f"repeated:   {metrics['no_of_nodes_repeated']}\n" +
                  f"{len(metrics['nodes_explored'])}\n")
            if len(node_env.action_trajectory) < len(CURRENT_BEST) or len(CURRENT_BEST) == 0:
                CURRENT_BEST = node_env.action_trajectory.copy()
            return node_env.action_trajectory
            #return node_env.action_trajectory, metrics, node_env

        if node_env._check_if_maxsteps():
            print("Maximal number of steps reached!")
            continue

        children = node_env.get_children()

        steps_in_depth += 1
        print(f"steps_in_depth = {steps_in_depth}")
        if steps_in_depth > max_steps_in_depth:
            steps_in_depth -= 1
            #return []

        # every child is None.
        if not [child for child in children if child is not None]:
            env_queue.pop(0)  # todo: Is this correct? Would pop a second time!

        # there is at least one feasible child which is not None.
        else:
            for action, child in reversed(list(enumerate(children))):
                if print_steps:
                    if metrics["no_of_nodes_discovered"] % 1000 == 0:
                        print(f'no_of_nodes_discovered: {metrics["no_of_nodes_discovered"]}')
                metrics['no_of_nodes_discovered'] += 1

                child_env = deepcopy(node_env)  # copy the environment to take a "virtual" step
                child_env.step(action)  # take a step in the "virtual" environment


                if flag:
                    print("###")
                    print(node_env.get_action_lookup_chars(action))
                    print(child_env.action_trajectory)
                    print(tuple(child_env.room_state.flatten()) not in metrics['nodes_explored'])
                    print(not env_is_in_envs(child_env, env_queue))

                if node_env.action_trajectory == ['L', 'L']:
                    if node_env.get_action_lookup_chars(action) == 'd':
                        print("-----------------------")
                        print(f"-- CURRENT_BEST: {CURRENT_BEST}")
                        print(f"-- child.env.at: {child_env.action_trajectory}")
                        print(child_env.room_state)
                        print(f"-- nodes explor: {len(metrics['nodes_explored'])}")
                        print(f"-- no in explor: {tuple(child_env.room_state.flatten()) not in metrics['nodes_explored']}")
                        print(f"-- !env in envs: {not env_is_in_envs(child_env, env_queue)}")
                        print("----------------------------------------------")


                if tuple(child_env.room_state.flatten()) not in metrics['nodes_explored'] and \
                        not env_is_in_envs(child_env, env_queue):
                    print(f"if ...... {child_env.action_trajectory}  {CURRENT_BEST} \n {child_env.room_state}")
                    env_queue.insert(0, child_env)
                    curr_best_sol = depth_first_search_recursive(child_env, metrics)

                    if 0 != len(curr_best_sol) < len(CURRENT_BEST) != 0:
                        CURRENT_BEST = curr_best_sol.copy()

                elif tuple(child_env.room_state.flatten()) in metrics['nodes_explored'] and \
                        len(child_env.action_trajectory) < len(CURRENT_BEST):
                    print(f"elif .... {child_env.action_trajectory}  {CURRENT_BEST} \n {child_env.room_state}")
                    env_queue.insert(0, child_env)
                    curr_best_sol = depth_first_search_recursive(child_env, metrics)

                    if 0 != len(curr_best_sol) < len(CURRENT_BEST) != 0:
                        CURRENT_BEST = curr_best_sol.copy()

                else:
                    metrics['no_of_nodes_repeated'] += 1


def env_is_in_envs(child_env, env_queue):
    return np.any([np.alltrue(b.room_state == child_env.room_state) for b in env_queue])

TEST = np.array([[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 2, 1, 1, 0, 0, 0],
 [0, 4, 1, 1, 1, 0, 0],
 [0, 1, 5, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]])