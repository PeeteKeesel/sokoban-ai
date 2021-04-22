import random
import time
import numpy as np

from gym_sokoban.envs.sokoban_env   import SokobanEnv, ACTION_LOOKUP_CHARS
from tests.testing_environment      import unittest
from src.algorithms.mcts            import MctsNode, Mcts

RANDOM_SEED = 0

# ================================================================
class TestMcts(unittest.TestCase):

    def setUp(self, dim_room=(6, 6), num_boxes=1, render_board=False, print_board=False, random_seed=RANDOM_SEED):
        self.mock_env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes)

        self.mock_env.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.mock_env.action_space.seed(random_seed)
        self.mock_env.reset()

        if print_board:
            print(f"\n---\nroom of size {dim_room} with {num_boxes} boxes and random_seed={random_seed}")
            print(self.mock_env.room_state)

        if render_board:
            self.mock_env.render_colored()

    def test_mcts(self):
        print("test_mcts")
        self.setUp(print_board=False, render_board=True)

        self.mock_env.step(1); self.mock_env.render_colored()
        self.mock_env.step(8); self.mock_env.render_colored()
        self.mock_env.step(5); self.mock_env.render_colored()
        self.mock_env.step(3); self.mock_env.render_colored()

        MctsNode(self.mock_env, self.mock_env.get_n_actions())



        pass

