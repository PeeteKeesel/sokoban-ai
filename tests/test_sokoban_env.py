import random
import numpy as np

from gym_sokoban.envs import MctsSokobanEnv
from gym_sokoban.envs.sokoban_env import SokobanEnv
from tests.testing_environment    import unittest
from utils import manhattan_distance
#$ from gym_sokoban.envs.sokoban_env import g

# ================================================================
class TestSokobanEnv(unittest.TestCase):

    def setUp(self, dim_room=(6, 6), num_boxes=3, print_board=False, render_board=False):
        self.mock_env = MctsSokobanEnv(dim_room=dim_room, num_boxes=num_boxes)

        self.mock_env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.mock_env.action_space.seed(RANDOM_SEED)
        self.mock_env.reset()

        if print_board:
            print(self.mock_env.room_state)

        if render_board:
            self.mock_env.render_colored()

    def test_state_after_action(self):
        self.setUp(dim_room=(6, 6), num_boxes=1, print_board=True)

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
        self.setUp(dim_room=(6, 6), num_boxes=1, print_board=True)

        self.assertTrue(np.alltrue(self.mock_env.room_state == INITIAL_ROOM_STATE))

        # expected successor state after action 'push up'
        child_expected_after_U = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 4, 1, 0],
                                          [0, 0, 0, 5, 1, 0],
                                          [0, 0, 0, 1, 1, 0],
                                          [0, 0, 0, 0, 0, 0]])
        # expected successor state after action 'move right'
        child_expected_after_r = np.array([[0, 0, 0, 0, 0, 0],
                                          [0, 1, 1, 1, 1, 0],
                                          [0, 0, 2, 1, 1, 0],
                                          [0, 0, 0, 4, 1, 0],
                                          [0, 0, 0, 1, 5, 0],
                                          [0, 0, 0, 0, 0, 0]])
        children_expected = [child_expected_after_U, child_expected_after_r]
        children_actual   = self.mock_env.get_children()

        children_not_None = [child for child in children_actual if child is not None]
        self.assertEqual(2,
                         len(children_not_None),
                         "length of elements which are not None should be 2.")
        self.assertTrue(np.array_equal(children_expected, children_not_None))

    def test_manhattan_distance(self):
        self.setUp()

        manh_dist_expect = 4
        manh_dist_actual = manhattan_distance(np.array([2, 3]), (5, 4))

        self.assertEqual(manh_dist_expect,
                         manh_dist_actual,
                         f"Manhattan distance between [2, 3] and [5, 4] should be {manh_dist_expect} but is {manh_dist_actual}")

        self.assertRaises(AssertionError,
                          manhattan_distance, np.array([2, 3, 4]), (5, 4))

    def test_manhattan_heuristic(self):
        self.setUp(print_board=True)

        manh_heur_expect = 1 + (1 + 1 + 2)
        manh_heur_actual = self.mock_env.manhattan_heuristic()

        self.assertEqual(manh_heur_expect,
                         manh_heur_actual,
                         f"Manhattan heuristic should be {manh_heur_expect} but is {manh_heur_actual} for \n{self.mock_env.room_state}")

    def test_in_corner(self):
        self.setUp(num_boxes=1, render_board=True)

        self.assertFalse(self.mock_env._in_corner())

        # 1st example
        self.mock_env.steps([8, 5, 5, 7, 2])
        self.assertTrue(self.mock_env._in_corner())

        # 2nd example
        self.setUp(num_boxes=1, render_board=True)
        self.mock_env.steps([1, 1, 7, 5, 4])
        self.assertTrue(self.mock_env._in_corner())

        # 3rd example
        self.setUp(num_boxes=1, render_board=True)
        self.mock_env.steps([1, 1, 8, 5, 3, 3])
        self.assertTrue(self.mock_env._in_corner())

    def test_deadlock_detection(self):
        self.setUp(num_boxes=1, render_board=True)

        # Test corner deadlock.
        self.mock_env.steps([8, 5, 5, 7])
        self.assertTrue(self.mock_env.deadlock_detection(actionToTake=2),
                        f"The room state after taking action 2 should be a deadlock.")

        # TODO: test other deadlocks e.g. simple deadlocks


    ##############################################################################
    # Test static-methods                                                        #
    ##############################################################################
    def test_get_actions_lookup_chars(self):
        self.setUp()
        self.assertEqual(
            self.mock_env.get_actions_lookup_chars([2, 3, 4, 5]),
            ["D", "L", "R", "u"],
            "The chars for actions [2,3,4,5] should be ['D', 'L', 'R', 'u]''")

        self.assertRaises(
            AssertionError,
            self.mock_env.get_actions_lookup_chars,
            [2, 3, 9, 5])

    def test_print_actions_as_chars(self):
        self.setUp()

        self.assertEqual(
            self.mock_env.print_actions_as_chars([2,3,4,5]),
            "DLRu",
            "The joined chars for actions [2,3,4,5] should be 'DLRu'"
        )

        self.assertEqual(
            self.mock_env.print_actions_as_chars([]),
            "",
            "The joined chars for actions [] should be ''"
        )


RANDOM_SEED = 0
INITIAL_ROOM_STATE = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 0, 2, 1, 1, 0],
                               [0, 0, 0, 4, 1, 0],
                               [0, 0, 0, 5, 1, 0],
                               [0, 0, 0, 0, 0, 0]])