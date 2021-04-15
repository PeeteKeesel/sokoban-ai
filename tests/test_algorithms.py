import random
import time
import numpy as np

from gym_sokoban.envs.sokoban_env   import SokobanEnv, ACTION_LOOKUP_CHARS
from tests.testing_environment      import unittest
from src.algorithms                 import depth_first_search    as dfs
from src.algorithms                 import breadth_first_search  as bfs
from src.algorithms                 import uniform_cost_search   as ucs
from src.algorithms                 import a_star_search         as astar

RANDOM_SEED = 0


# ================================================================
class TestAlgorithms(unittest.TestCase):

    # def __init__(self, methodName: str = ...):
    #     super().__init__(methodName)
    #     self.mock_env = None

    def setUp(self, dim_room=(6, 6), num_boxes=3, print_board=False, random_seed=RANDOM_SEED):
        self.mock_env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes)

        self.mock_env.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.mock_env.action_space.seed(random_seed)
        self.mock_env.reset()

        if print_board:
            print(f"\n---\nroom of size {dim_room} with {num_boxes} boxes and random_seed={random_seed}")
            print(self.mock_env.room_state)

    def test_optimal(self):
        self.setUp()

        o_10x10 = ['d', 'r', 'd', 'r', 'r', 'u', 'u', 'l', 'u', 'r', 'u', 'r', 'U', 'r', 'u', 'L', 'r', 'd', 'l', 'l', 'l', 'l',
         'u', 'r', 'R', 'l', 'l', 'd', 'r', 'r', 'd', 'l', 'd', 'r', 'd', 'l', 'L', 'r', 'r', 'd', 'd', 'l', 'l', 'l',
         'u', 'r', 'R', 'l', 'l', 'd', 'r', 'r', 'r', 'U', 'l', 'l', 'u', 'r', 'u', 'r', 'u', 'r', 'u', 'r', 'u', 'L',
         'r', 'd', 'l', 'l', 'l', 'l', 'u', 'r', 'R', 'R', 'l', 'l', 'l', 'd', 'r', 'r', 'r', 'r', 'U', 'l', 'l', 'l',
         'l', 'd', 'r', 'r', 'r', 'd', 'l', 'l', 'd', 'r', 'D', 'l', 'l', 'd', 'r', 'd', 'r', 'U', 'l', 'l', 'u', 'r',
         'R', 'l', 'l', 'd', 'r', 'r', 'd', 'l', 'l', 'l']

        o_6x6 = ['d', 'L', 'r', 'u', 'L', 'r', 'd', 'l', 'd', 'd', 'r', 'U', 'l', 'u', 'u', 'L']
        o_6x6_bfs = ['L', 'L', 'R', 'R', 'D', 'L', 'D', 'D', 'R', 'U']

        o_8x8 = ['r', 'u', 'u', 'l', 'l', 'D', 'l', 'd', 'R', 'l', 'u', 'r', 'u', 'r', 'r', 'd', 'd', 'L', 'U', 'l', 'l', 'd', 'R', 'l', 'u', 'r', 'u', 'R', 'l', 'd', 'r', 'r', 'd', 'L', 'r', 'u', 'U', 'l', 'l', 'd', 'l', 'd', 'R', 'l', 'u', 'r', 'r', 'r', 'u', 'U', 'd', 'l', 'l', 'd', 'r', 'r', 'd', 'L', 'r', 'u', 'l', 'l', 'u', 'r', 'r', 'u', 'U', 'l', 'D', 'r', 'd', 'd', 'l', 'l', 'l', 'd', 'R', 'l', 'u', 'r', 'r', 'r', 'u', 'u', 'l', 'D', 'r', 'd', 'd', 'L', 'r', 'u', 'u', 'l', 'l', 'd', 'R', 'd', 'r', 'U', 'U', 'U', 'l', 'u', 'R']
        o_8x8_ucs = ['U', 'D', 'L', 'L', 'U', 'R', 'U', 'R', 'L', 'D', 'D', 'R', 'R', 'U', 'L', 'R', 'U', 'L', 'D', 'D', 'L', 'L', 'U', 'R', 'R', 'U', 'R', 'U', 'U', 'L', 'D', 'D', 'L', 'D', 'D', 'R', 'R', 'U', 'L', 'R', 'U', 'L', 'D', 'D', 'L', 'L', 'U', 'R', 'D', 'R', 'U', 'R', 'U', 'U', 'L', 'D', 'R', 'D', 'L', 'U', 'U', 'U', 'R', 'L', 'D', 'D', 'L', 'D']
        o_10x10_ucs = ['R', 'R', 'U', 'R', 'D', 'L', 'U', 'U', 'R', 'R', 'U', 'L', 'D', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'R', 'R', 'R', 'U', 'L', 'U', 'U', 'R', 'U', 'R', 'U', 'U', 'L', 'L', 'L', 'D', 'R', 'R', 'L', 'D', 'D', 'R', 'D', 'L', 'L', 'L', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'U', 'L', 'U', 'U', 'R', 'U', 'R', 'R', 'D', 'L', 'L', 'R', 'D', 'L', 'R', 'U', 'U', 'L', 'L', 'L', 'D', 'R', 'R', 'L', 'D', 'R', 'U', 'U', 'R', 'R', 'D', 'L', 'L', 'D', 'L', 'D', 'R', 'D', 'L', 'R', 'U', 'U', 'R', 'U', 'U', 'L', 'L', 'D', 'D', 'D', 'D', 'R', 'D', 'L', 'D', 'L', 'L', 'U', 'U', 'R', 'R', 'D', 'R', 'U']
        sol10x10_2 = ['R', 'L', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'L', 'R', 'U', 'U', 'R', 'U', 'L', 'D', 'L', 'D', 'D', 'R', 'D', 'D', 'L', 'L', 'L', 'U', 'U', 'R', 'L', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'U', 'L', 'D', 'R', 'U', 'U', 'R', 'U', 'U', 'L', 'L', 'L', 'D', 'R', 'R', 'L', 'D', 'D', 'R', 'D', 'D', 'D', 'L', 'U', 'R', 'U', 'U', 'L', 'U', 'U', 'R', 'U', 'R', 'R', 'D', 'L', 'L', 'U', 'L', 'L', 'D', 'R', 'D', 'D', 'D', 'R', 'U', 'U', 'R', 'U', 'U', 'L', 'D', 'D', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'R', 'R', 'R', 'U', 'L']

        o = sol10x10_2

        for a in o:
            self.mock_env.render()
            time.sleep(0.1)
            i = list(ACTION_LOOKUP_CHARS.keys())[list(ACTION_LOOKUP_CHARS.values()).index(a)]
            self.mock_env.step(i)
        self.mock_env.render()
        time.sleep(30)

    # def test_set_children(self):
    #     self.setUp()
    #     searches = Searches(self.mock_env)
    #
    #     self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))
    #     self.assertTrue(np.alltrue(searches.env.room_state == INITIAL_ROOM_STATE))
    #     print(self.mock_env.room_state)
    #
    #     # expected successor state after action 'move right'
    #     expected_succ_after_r = np.array([[0, 0, 0, 0, 0, 0],
    #                                       [0, 1, 1, 1, 1, 0],
    #                                       [0, 0, 2, 4, 1, 0],
    #                                       [0, 0, 0, 5, 1, 0],
    #                                       [0, 0, 0, 1, 1, 0],
    #                                       [0, 0, 0, 0, 0, 0]])
    #     # expected successor state after action 'push up'
    #     expected_succ_after_U = np.array([[0, 0, 0, 0, 0, 0],
    #                                       [0, 1, 1, 1, 1, 0],
    #                                       [0, 0, 2, 1, 1, 0],
    #                                       [0, 0, 0, 4, 1, 0],
    #                                       [0, 0, 0, 1, 5, 0],
    #                                       [0, 0, 0, 0, 0, 0]])
    #
    #
    #     # children = self.mock_env.set_children()
    #     children = searches.set_children()
    #     self.assertEqual(len(children), len(ACTION_LOOKUP.keys()), "Total length should be 9")
    #
    #     print(children)
    #     feasible_children = [child for child in children if child is not None]
    #     self.assertEqual(len(feasible_children), 2, "length of elements not None should be 2")
    #     self.assertTrue(np.array_equal([expected_succ_after_r, expected_succ_after_U], feasible_children))

    def test_depth_first_search(self):
        print("test_depth_first_search")
        self.setUp(dim_room=(5, 5), num_boxes=1, print_board=True)

        start = time.time()
        metrics, node_env = dfs(self.mock_env, print_steps=True)
        end = time.time()

        print(f'runtime: {round(end - start, 4)} seconds')

    def test_breadth_first_search(self):
        print("test_breadth_first_search")
        self.setUp(print_board=True)

        start = time.time()
        metrics, node_env = bfs(self.mock_env, print_steps=True)
        end = time.time()

        print(f'runtime: {round(end - start, 4)} seconds')

    def test_uniform_cost_search(self):
        print("test_uniform_cost_search")
        self.setUp(dim_room=(5, 5), num_boxes=1, print_board=True)

        start = time.time()
        _, _ = ucs(self.mock_env, print_steps=True)
        end = time.time()

        print(f'runtime: {round(end - start, 4)} seconds')

    def test_a_start_search(self):
        print("test_a_start_search")

        self.setUp(dim_room=(5, 5), num_boxes=1, print_board=True)

        start = time.time()
        _, _ = astar(self.mock_env, print_steps=True)
        end = time.time()

        print(f"runtime: {round(end - start, 4)} seconds")


    def test_algo_on_multiple_levels(self):
        print("test_algo_on_multiple_levels")
        no_of_games_to_solve = 20

        total_time = 0
        for seed in range(0, no_of_games_to_solve):
            self.setUp(dim_room=(6, 6), num_boxes=2, print_board=True, random_seed=seed)

            start = time.time()
            _, _ = astar(self.mock_env, print_steps=True)
            end = time.time()

            current_time = end - start
            print(f"runtime: {round(current_time, 4)} seconds")
            total_time += current_time

        print(f"total runtime for {no_of_games_to_solve} levels: {round(total_time, 4)} seconds.")

    def test_temp(self):
        self.setUp()

        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 5, 0],
                  [0, 0, 2, 1, 1, 0],
                  [0, 0, 0, 4, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0]])

        self.mock_env.step(8)
        self.mock_env.step(5)
        self.mock_env.step(5)
        self.mock_env.step(5)
        print(self.mock_env.room_state)

        searches = Searches(self.mock_env)
        searches.set_children()

        for child in searches.current_node.children:
            if child is not None:
                print(f"action:{child.action}  room_state: \n{child.room_state}")

    def test_depth_first_search_v2(self):
        # initialize Tree with start game state as root
        self.setUp()

        searches = Searches(self.mock_env)
        expected_opt_sol = ['U', 'r', 'u', 'L']
        discovered = set()
        terminated = False

        print(searches.env.room_state)
        steps = 0

        searches.DFS_search(discovered=discovered, terminated=terminated, steps=steps)
        print(steps)

        self.assertEqual(expected_opt_sol,
                         searches.solution,
                         "Optimal solution should be ['U', 'r', 'u', 'L']")
        # g = Graph({self.mock_env.room_state: []})
        # t = Node(self.mock_env.room_state)

        # implement in DFS
        # iterate until 1 sol found or 2 max number iter
        # 1. find all possible actions for each positions (board state)
            # add the next-board-states as childs to the graph and go to the left node
        # 2. do this only on the left side until finished or max_steps
            # if max_steps go right and go until finished or max_steps in depth again
            # if no children is there anymore and it was never finished -> STOP

        # self.search.depth_first_search()


        # mock_env.render('format')

    def test_add_children_nodes(self):
     self.setUp()

     # self.search.add_children_nodes()


    # def test__get_all_feasible_actions(self):
    #     self.setUp()
    #     print(self.mock_env.room_state)
    #
    #     actions = self.mock_env._get_all_feasible_actions()
    #     print(actions)
    #

    #
    # def test_reverse_step(self):
    #     self.setUp()
    #
    #     # move right
    #     self.mock_env.step(8)
    #     print(self.mock_env.room_state)
    #
    #     self.mock_env.reverse_step(8)
    #     print(self.mock_env.room_state)



INITIAL_ROOM_STATE_8x8 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 4, 5, 0, 0],
                                   [0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 1, 2, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0]])
