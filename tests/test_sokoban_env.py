import random
import numpy as np

from gym_sokoban.envs.sokoban_env import SokobanEnv
from tests.testing_environment    import unittest

RANDOM_SEED = 0

# ================================================================
class TestSokobanEnv(unittest.TestCase):

    def setUp(self, dim_room=(6, 6), num_boxes=3, print_board=False):
        self.mock_env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes)

        self.mock_env.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.mock_env.action_space.seed(RANDOM_SEED)
        self.mock_env.reset()

        if print_board:
            print(self.mock_env.room_state)

    def test_manhattan_distance(self):
        self.setUp()

        manh_dist_expect = 4
        manh_dist_actual = self.mock_env.manhatten_distance(np.array([2, 3]), (5, 4))

        self.assertEqual(manh_dist_expect,
                         manh_dist_actual,
                         f"Manhattan distance between [2, 3] and [5, 4] should be {manh_dist_expect} but is {manh_dist_actual}")

        self.assertRaises(AssertionError,
                          self.mock_env.manhatten_distance, np.array([2, 3, 4]), (5, 4))

    def test_manhattan_heuristic(self):
        self.setUp(print_board=False)

        manh_heur_expect = 1 + (1 + 1 + 2)
        manh_heur_actual = self.mock_env.manhattan_heuristic()

        self.assertEqual(manh_heur_expect,
                         self.mock_env.manhattan_heuristic(),
                         f"Manhattan heuristic should be {manh_heur_expect} but is {manh_heur_actual} for \n{self.mock_env.room_state}")
