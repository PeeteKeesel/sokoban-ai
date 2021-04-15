from copy               import deepcopy
from gym_sokoban.envs   import SokobanEnv

def depth_first_search(env: SokobanEnv, print_steps: bool=None):
    """
    Traverses the given {@env} in a depth first search way until a termination condition is met.

    Time: O(|V| + |E|)
    Space: O(|V|)
    """

    metrics = {
        'no_of_nodes_discovered': 0,    # the total number of discovered nodes. Including repeated ones.
        'no_of_nodes_repeated': 0,      # the number of a times nodes got discovered repeatedly.
        'nodes_explored': set(),        # the set of all discovered nodes excluding duplications.
        'environemnts': set(),          # this saves the environment of the nodes.
    }

    if env._check_if_done():
        return metrics, env

    env_queue = [env]  # this serves as the stack for the environments.

    while True:
        if not env_queue:
            print(metrics)
            raise Exception('depth_first_search(): Solution NOT FOUND! Empty environment queue.')

        node_env = env_queue.pop(0)
        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))

        if node_env._check_if_all_boxes_on_target():
            print("------------------------------------------------------------------------------------\n" +
                  f"depth_first_search(): Solution FOUND! Got {len(node_env.action_trajectory)} steps\n" +
                  f"{node_env.action_trajectory}\n" +
                  f"discovered: {metrics['no_of_nodes_discovered']}\n" +
                  f"repeated:   {metrics['no_of_nodes_repeated']}\n" +
                  f"{len(metrics['nodes_explored'])}\n" +
                  f"{node_env.room_state}")
            return metrics, node_env

        if node_env._check_if_maxsteps():
            print("Maximal number of steps reached!")
            continue

        children = node_env.get_children()

        if not [child for child in children if child is not None]:
            env_queue.pop(0) # todo: Is this correct? Would pop a second time!

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
                    env_queue.insert(0, child_env)

                else:
                    metrics['no_of_nodes_repeated'] += 1
