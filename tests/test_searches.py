import random
import numpy as np

from gym_sokoban.envs.sokoban_env import SokobanEnv, Search, ACTION_LOOKUP
from tests.testing_environment import unittest


RANDOM_SEED = 0

class Graph:
    def __init__(self, G: dict):
        if G is None:
            G = {}
        self.G = G

class Node:
    def __init__(self, data):
        """ Initialize a Node object.

        Arguments:
            data - Data about the Node of any type.
        """
        self.data = data
        self.children = []

# ================================================================
class TestSearches(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.mock_env = None

    def setUp(self):
        self.mock_env = SokobanEnv(dim_room=(6, 6), num_boxes=1)

        self.mock_env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.mock_env.action_space.seed(RANDOM_SEED)
        self.mock_env.reset()

        self.search = Search(self.mock_env, Node(self.mock_env.room_state))

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

    def test_successors(self):
        self.setUp()
        self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))
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

        self.assertEqual(len(self.mock_env.successors()), 2, "length should be 2")
        self.assertTrue(np.array_equal([expected_succ_after_r, expected_succ_after_U], self.mock_env.successors()))


    def test_depth_first_search(self):
        # initialize Tree with start game state as root
        self.setUp()

        # g = Graph({self.mock_env.room_state: []})
        # t = Node(self.mock_env.room_state)

        # implement in DFS
        # iterate until 1 sol found or 2 max number iter
        # 1. find all possible actions for each positions (board state)
            # add the next-board-states as childs to the graph and go to the left node
        # 2. do this only on the left side until finished or max_steps
            # if max_steps go right and go until finished or max_steps in depth again
            # if no children is there anymore and it was never finished -> STOP

        self.search.depth_first_search()


        # mock_env.render('format')

    def test_add_children_nodes(self):
     self.setUp()

     self.search.add_children_nodes()


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