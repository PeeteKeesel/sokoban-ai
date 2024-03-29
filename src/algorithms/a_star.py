from copy import deepcopy
from utils import \
    get_search_algorithm_results, \
    MESSAGE_SOLUTION_FOUND, MESSAGE_TIME_LIMIT_REACHED, ALGORITHM_NAME_A_STAR, \
    METRICS_SCELETON
from time import time


def a_star_search(env, time_limit: int, metrics: dict=None, print_steps: bool=None):
    """Informed search algorithm / Best first search.
    b = branching factor
    d = depth

    Time: O(|E|) = (b^d)
    Space: O(|V|) = O(b^d)

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

    env_queue = PriorityQueueAStar()  # initialize a priority queue for A star
    env_queue.push(0, env.manhattan_heuristic(), env)
    metrics['no_of_nodes_discovered'] += 1

    while True:
        start_time = time()

        if not env_queue:
            print(metrics)
            raise Exception(f"{ALGORITHM_NAME_A_STAR}(): Solution NOT FOUND! "
                            f"Empty environment queue.")

        node_env_cost_total, node_env_cost_actual, node_env = env_queue.pop()   # get the environment with the lowest cost

        if current_time >= time_limit:
            return get_search_algorithm_results(ALGORITHM_NAME_A_STAR,
                                                node_env, metrics,
                                                MESSAGE_TIME_LIMIT_REACHED)

        if node_env.all_boxes_on_target():
            return get_search_algorithm_results(ALGORITHM_NAME_A_STAR,
                                                node_env, metrics,
                                                MESSAGE_TIME_LIMIT_REACHED)

        if node_env.max_steps_reached():
            #print("Maximal number of steps reached!")
            continue

        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))
        children = node_env.get_children()

        if not [child for child in children if child is not None]:  # todo: does it make sense?
            env_queue.pop()  # No children exist which are not None

        else:
            for action, child in enumerate(children):
                if print_steps:
                    if metrics["no_of_nodes_discovered"] % 10_000 == 0:
                        print(f'no_of_nodes_discovered: {metrics["no_of_nodes_discovered"]}')

                child_env       = deepcopy(node_env)      # copy the environment to take a "virtual" step
                _, reward, _, _ = child_env.step(action)  # take a step in the "virtual" environment
                metrics['no_of_nodes_discovered'] += 1

                if tuple(child_env.room_state.flatten()) not in metrics['nodes_explored']:
                    if child_env not in env_queue.queue:
                        env_queue.push(node_env_cost_actual + reward,
                                       child_env.manhattan_heuristic(),
                                       child_env)
                    else: # todo: this checks if cost can be made lower. should instead be check for being higher?
                        env_queue.update_cost(node_env_cost_actual + reward,
                                              child_env.manhattan_heuristic(),
                                              child_env)  # todo: kann das nicht ein rewards aus einer ganz anderen trajektorie sein?
                        metrics['no_of_nodes_repeated'] += 1
                else:
                    metrics['no_of_nodes_repeated'] += 1

        # Update time and action trajectory.
        current_time += time() - start_time
        metrics['time'] = current_time
        metrics['action_traj'] = \
            node_env.get_actions_lookup_chars(node_env.action_trajectory)


"""Ordered queue by cost of being in a particular room state."""
class PriorityQueueAStar:
    def __init__(self):
        """
        A PriorityQueue object consists of a queue and board. It saves the manhattan heuristic for each env.
            queue - This serves as the priority queue. It contains tuples consisting of a Sokoban room state
                    and the corresponding cost for this room state.
                    [(cost_1, state_1), (cost_2, state_2), ... (cost_n, state_n)]
                    for which cost_1 < cost_2 < ... < cost_n
            board - This saves all pushed Sokoban room states at the index with the corresponding cost
                    as the value.
                    {room_state_1: cost_1, ..., room_state_n: cost_n}
        """
        self.queue  = []
        self.boards = {}


    def push(self, cost_heur, cost_actual, env):
        """
        If the given {@cost} ist smaller than any of the costs in the queue of the object
        then add this {@cost} and its corresponding {@room_state} to the queue. It will be
        added at the index for which the next board in the queue has a higher and the previous
        board has a lower cost.

        Arguments:
            cost_heur    (float)       - The cost for the the given room_state obtained by the heuristic.
            cost_actual  (float)       - The actual received cost for the the given room_state.
            env          (SokobanEnv)  - A state of the Sokoban Board.
        """
        if tuple(env.room_state.flatten()) in self.boards:
            # raise Exception("PriorityQueue.push(): ERROR: Repeated board was tried to being added to the priority queue.")
            # print("PriorityQueue.push(): ERROR: Repeated board was tried to being added to the priority queue.")
            return

        cost_total = cost_actual + cost_heur

        self.boards[tuple(env.room_state.flatten())] = cost_total

        # Check if the cost is smaller than any existing costs in the queue.
        add_to_queue = True
        for i, (cost_heur, cost_actual, board_env) in enumerate(self.queue):
            if cost_total <= cost_heur:
                self.queue.insert(i, (cost_total, cost_actual, env))
                add_to_queue = False
                break

        # if the cost is not smaller than any existing costs in the queue, add it to the end.
        if add_to_queue:
            self.queue.append((cost_total, cost_actual, env))


    def pop(self, index=0):
        cost_heur, cost_actual, env = self.queue.pop(index)  # delete the board and the corresponding costs from the queue.
        del self.boards[tuple(env.room_state.flatten())]     # delete the room state from the boards.

        assert len(self.queue) == len(self.boards.keys())

        return cost_heur, cost_actual, env


    def update_cost(self, cost_heur, cost_actual, env):
        cost_total = cost_actual + cost_heur

        if cost_total < self.boards[tuple(env.room_state.flatten())]:
            for i, (cost_heur, cost_actual, board_env) in enumerate(self.queue):
                if tuple(env.room_state.flatten()) == tuple(board_env.flatten()):  # found the room state in the queue.
                    self.queue[i][0] = (cost_total, cost_actual, env)              # update the cost in the queue.
                    self.boards[tuple(env.room_state.flatten())] = cost_total      # update the cost value in boards.


    def __bool__(self):
        return len(self.queue) > 0 and len(self.boards.keys()) > 0


    def __contains__(self, env):
        return tuple(env.room_state.flatten()) in self.boards.keys()


    def __len__(self):
        return len(self.queue)
