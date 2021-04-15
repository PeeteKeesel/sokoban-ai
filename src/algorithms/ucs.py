from copy               import deepcopy
from gym_sokoban.envs   import SokobanEnv

def uniform_cost_search(env: SokobanEnv, print_steps: bool=None):
    """Uninformed search algorithm."""

    metrics = {
        'no_of_nodes_discovered': 0,    # the total number of discovered nodes. Including repeated ones.
        'no_of_nodes_repeated': 0,      # the number of a times nodes got discovered repeatedly.
        'nodes_explored': set(),        # the set of all discovered nodes excluding duplications.
        'environemnts': set(),          # this saves the environment of the nodes.
    }

    if env._check_if_done():
        return metrics, env

    env_queue = PriorityQueueUcs()  # initialize a priority queue for ucs
    env_queue.push(0, env)
    metrics['no_of_nodes_discovered'] += 1
    while True:         # while the queue is not empty

        if not env_queue:
            raise Exception('uniform_cost_search(): Solution NOT FOUND! Empty environment queue.')

        node_env_cost, node_env = env_queue.pop()   # get the environment with the lowest cost

        if node_env._check_if_all_boxes_on_target():
            print("------------------------------------------------------------------------------------\n" +
                  f"uniform_cost_search(): Solution FOUND! Got {len(node_env.action_trajectory)} steps\n" +
                  f"{node_env.action_trajectory}\n" +
                  f"discovered: {metrics['no_of_nodes_discovered']}\n" +
                  f"repeated:   {metrics['no_of_nodes_repeated']}\n" +
                  f"{len(metrics['nodes_explored'])}\n" +
                  f"{node_env.room_state}")
            return metrics, node_env

        if node_env._check_if_maxsteps():
            #print("Maximal number of steps reached!")
            continue

        metrics['nodes_explored'].add(tuple(node_env.room_state.flatten()))
        children = node_env.get_children()

        if not [child for child in children if child is not None]:
            env_queue.pop()  # No children exist which are not None

        else:
            for action, child in enumerate(children):
                if print_steps:
                    if metrics["no_of_nodes_discovered"] % 10000 == 0:
                        print(f'no_of_nodes_discovered: {metrics["no_of_nodes_discovered"]}')

                child_env       = deepcopy(node_env)      # copy the environment to take a "virtual" step
                _, reward, _, _ = child_env.step(action)  # take a step in the "virtual" environment
                metrics['no_of_nodes_discovered'] += 1

                if tuple(child_env.room_state.flatten()) not in metrics['nodes_explored']:
                    if child_env not in env_queue.queue:
                        env_queue.push(node_env_cost + reward, child_env)
                    else:
                        env_queue.update_cost(node_env_cost + reward, child_env)  # todo: kann das nicht ein rewards aus einer ganz anderen trajektorie sein?
                        metrics['no_of_nodes_repeated'] += 1
                else:
                    metrics['no_of_nodes_repeated'] += 1


"""Ordered queue by cost of being in a particular room state."""
class PriorityQueueUcs:
    def __init__(self):
        """
        A PriorityQueue object consists of a queue and board. It saves the cost/reward for each env.
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


    def push(self, cost, env):
        """
        If the given {@cost} ist smaller than any of the costs in the queue of the object
        then add this {@cost} and its corresponding {@room_state} to the queue. It will be
        added at the index for which the next board in the queue has a higher and the previous
        board has a lower cost.

        Arguments:
            cost    (float)       - The cost for the the given room_state
            env     (SokobanEnv)  - A state of the Sokoban Board.
        """
        if tuple(env.room_state.flatten()) in self.boards:
            # raise Exception("PriorityQueue.push(): ERROR: Repeated board was tried to being added to the priority queue.")
            # print("PriorityQueue.push(): ERROR: Repeated board was tried to being added to the priority queue.")
            return

        self.boards[tuple(env.room_state.flatten())] = cost

        # Check if the cost is smaller than any existing costs in the queue.
        add_to_queue = True
        for i, (board_env_cost, board_env) in enumerate(self.queue):
            if cost <= board_env_cost:
                self.queue.insert(i, (cost, env))
                add_to_queue = False
                break

        # If the cost is not smaller than any existing costs in the queue, add it to the end.
        if add_to_queue:
            self.queue.append((cost, env))


    def pop(self, index=0):
        # print(f"  {len(self.queue)}\n --------------------START------------------")
        # for i in range(len(self.queue)):
        #     print(self.queue[i][1].room_state)
        # print(f" --------------------END--------------------")

        cost, env = self.queue.pop(index)                   # delete the board and the corresponding cost from the queue.
        del self.boards[tuple(env.room_state.flatten())]    # delete the room state from the boards.

        # print(f"{len(self.queue)} ?= {len(self.boards.keys())}")
        assert len(self.queue) == len(self.boards.keys())

        return cost, env


    def update_cost(self, cost, env):
        if cost < self.boards[tuple(env.room_state.flatten())]:
            for i, (board_env_cost, board_env) in enumerate(self.queue):
                if tuple(env.room_state.flatten()) == tuple(board_env.flatten()):
                    self.queue[i][0] = cost                              # update the cost in the queue.
                    self.boards[tuple(env.room_state.flatten())] = cost  # update the cost value in boards.


    def __bool__(self):
        return len(self.queue) > 0 and len(self.boards.keys()) > 0


    def __contains__(self, env):
        return tuple(env.room_state.flatten()) in self.boards.keys()


    def __len__(self):
        return len(self.queue)
