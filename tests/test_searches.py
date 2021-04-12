import random
import time
import numpy as np

from gym_sokoban.envs.sokoban_env import SokobanEnv, ACTION_LOOKUP, ACTION_LOOKUP_CHARS, \
    depth_first_search, breadth_first_search
from tests.testing_environment import unittest

RANDOM_SEED = 0


# ================================================================
class TestSearches(unittest.TestCase):

    # def __init__(self, methodName: str = ...):
    #     super().__init__(methodName)
    #     self.mock_env = None

    def setUp(self):
        self.mock_env = SokobanEnv(dim_room=(6, 6), num_boxes=3)

        self.mock_env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.mock_env.action_space.seed(RANDOM_SEED)
        self.mock_env.reset()

        #self.search = Search(self.mock_env, Node(self.mock_env.room_state))

    def test_state_after_action(self):
        self.setUp()
        print(self.mock_env.room_state)
        print(self.mock_env.state_after_action(1))

        self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))

        # Pushes
        # push up -> change
        self.assertTrue(self.mock_env.state_after_action(1)['state_changed'])
        expected_state_after_action = np.array([[0, 0, 0, 0, 0, 0],
                                                [0, 1, 1, 1, 1, 0],
                                                [0, 0, 2, 4, 1, 0],
                                                [0, 0, 0, 5, 1, 0],
                                                [0, 0, 0, 1, 1, 0],
                                                [0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.alltrue(expected_state_after_action == self.mock_env.state_after_action(1)['new_state']))

        # push down -> no change
        self.assertFalse(self.mock_env.state_after_action(2)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(2)['new_state']))

        # push left -> no change
        self.assertFalse(self.mock_env.state_after_action(3)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(3)['new_state']))

        # push right -> no change
        self.assertFalse(self.mock_env.state_after_action(4)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(4)['new_state']))

        # Moves
        # move up -> no change
        self.assertFalse(self.mock_env.state_after_action(5)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(5)['new_state']))

        # move down -> no change
        self.assertFalse(self.mock_env.state_after_action(6)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(6)['new_state']))

        # move left -> no change
        self.assertFalse(self.mock_env.state_after_action(7)['state_changed'])
        self.assertTrue(np.alltrue(self.mock_env.room_state == self.mock_env.state_after_action(7)['new_state']))

        # move right -> change
        self.assertTrue(self.mock_env.state_after_action(8)['state_changed'])
        expected_state_after_action = np.array([[0, 0, 0, 0, 0, 0],
                                                [0, 1, 1, 1, 1, 0],
                                                [0, 0, 2, 1, 1, 0],
                                                [0, 0, 0, 4, 1, 0],
                                                [0, 0, 0, 1, 5, 0],
                                                [0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.alltrue(expected_state_after_action == self.mock_env.state_after_action(8)['new_state']))

    def test_get_children(self):
        self.setUp()

        self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))
        print(self.mock_env.room_state)

        # expected successor state after action 'move right'
        child_expected_after_r = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 4, 1, 0],
                                          [0, 0, 0, 5, 1, 0],
                                          [0, 0, 0, 1, 1, 0],
                                          [0, 0, 0, 0, 0, 0]])
        # expected successor state after action 'push up'
        child_expected_after_U = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 1, 1, 0],
                                          [0, 0, 0, 4, 1, 0],
                                          [0, 0, 0, 1, 5, 0],
                                          [0, 0, 0, 0, 0, 0]])
        children_expected = [child_expected_after_r, child_expected_after_U]
        children_actual   = self.mock_env.get_children()

        children_not_None = [child for child in children_actual if child is not None]
        self.assertEqual(len(children_not_None), 2, "length of elements not None should be 2")
        self.assertTrue(np.array_equal(children_expected, children_not_None))

    def test_optimal(self):
        self.setUp()

        o_10x10 = ['d', 'r', 'd', 'r', 'r', 'u', 'u', 'l', 'u', 'r', 'u', 'r', 'U', 'r', 'u', 'L', 'r', 'd', 'l', 'l', 'l', 'l',
         'u', 'r', 'R', 'l', 'l', 'd', 'r', 'r', 'd', 'l', 'd', 'r', 'd', 'l', 'L', 'r', 'r', 'd', 'd', 'l', 'l', 'l',
         'u', 'r', 'R', 'l', 'l', 'd', 'r', 'r', 'r', 'U', 'l', 'l', 'u', 'r', 'u', 'r', 'u', 'r', 'u', 'r', 'u', 'L',
         'r', 'd', 'l', 'l', 'l', 'l', 'u', 'r', 'R', 'R', 'l', 'l', 'l', 'd', 'r', 'r', 'r', 'r', 'U', 'l', 'l', 'l',
         'l', 'd', 'r', 'r', 'r', 'd', 'l', 'l', 'd', 'r', 'D', 'l', 'l', 'd', 'r', 'd', 'r', 'U', 'l', 'l', 'u', 'r',
         'R', 'l', 'l', 'd', 'r', 'r', 'd', 'l', 'l', 'l']

        o_6x6 = ['d', 'L', 'r', 'u', 'L', 'r', 'd', 'l', 'd', 'd', 'r', 'U', 'l', 'u', 'u', 'L']

        o_8x8 = ['r', 'u', 'u', 'l', 'l', 'D', 'l', 'd', 'R', 'l', 'u', 'r', 'u', 'r', 'r', 'd', 'd', 'L', 'U', 'l', 'l', 'd', 'R', 'l', 'u', 'r', 'u', 'R', 'l', 'd', 'r', 'r', 'd', 'L', 'r', 'u', 'U', 'l', 'l', 'd', 'l', 'd', 'R', 'l', 'u', 'r', 'r', 'r', 'u', 'U', 'd', 'l', 'l', 'd', 'r', 'r', 'd', 'L', 'r', 'u', 'l', 'l', 'u', 'r', 'r', 'u', 'U', 'l', 'D', 'r', 'd', 'd', 'l', 'l', 'l', 'd', 'R', 'l', 'u', 'r', 'r', 'r', 'u', 'u', 'l', 'D', 'r', 'd', 'd', 'L', 'r', 'u', 'u', 'l', 'l', 'd', 'R', 'd', 'r', 'U', 'U', 'U', 'l', 'u', 'R']

        o_6x6_bfs = ['L', 'L', 'R', 'R', 'D', 'L', 'D', 'D', 'R', 'U']

        o = o_6x6_bfs

        for a in o:
            self.mock_env.render()
            time.sleep(0.3)
            i = list(ACTION_LOOKUP_CHARS.keys())[list(ACTION_LOOKUP_CHARS.values()).index(a)]
            self.mock_env.step(i)
        self.mock_env.render()
        time.sleep(30)

    def test_depth_first_search(self):
        self.setUp()
        print(self.mock_env.room_state)
        print(self.mock_env.print_room_state_using_format())

        start = time.time()
        metrics, node_env = depth_first_search(self.mock_env, print_steps=True)
        end = time.time()

        print(f'runtime: {round(end - start, 4)} seconds')


    def test_set_children(self):
        self.setUp()
        searches = Searches(self.mock_env)

        self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))
        self.assertTrue(np.alltrue(searches.env.room_state == INITIAL_ROOM_STATE))
        print(self.mock_env.room_state)

        # expected successor state after action 'move right'
        expected_succ_after_r = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 4, 1, 0],
                                          [0, 0, 0, 5, 1, 0],
                                          [0, 0, 0, 1, 1, 0],
                                          [0, 0, 0, 0, 0, 0]])
        # expected successor state after action 'push up'
        expected_succ_after_U = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 1, 1, 0],
                                          [0, 0, 0, 4, 1, 0],
                                          [0, 0, 0, 1, 5, 0],
                                          [0, 0, 0, 0, 0, 0]])


        # children = self.mock_env.set_children()
        children = searches.set_children()
        self.assertEqual(len(children), len(ACTION_LOOKUP.keys()), "Total length should be 9")

        print(children)
        feasible_children = [child for child in children if child is not None]
        self.assertEqual(len(feasible_children), 2, "length of elements not None should be 2")
        self.assertTrue(np.array_equal([expected_succ_after_r, expected_succ_after_U], feasible_children))

    def test_breadth_first_search(self):
        self.setUp()

        opt_sol_expected = ['U', 'r', 'u', 'L']
        start = time.time()
        metrics, node_env = breadth_first_search(self.mock_env, print_steps=True)
        end = time.time()

        print(f'runtime: {round(end - start, 4)} seconds')

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

INITIAL_ROOM_STATE = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 0, 2, 1, 1, 0],
                               [0, 0, 0, 4, 1, 0],
                               [0, 0, 0, 5, 1, 0],
                               [0, 0, 0, 0, 0, 0]])

INITIAL_ROOM_STATE_8x8 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 4, 5, 0, 0],
                                   [0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 1, 2, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0]])
