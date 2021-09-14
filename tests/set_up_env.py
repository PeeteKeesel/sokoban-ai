import random
import numpy as np

from algorithms.mcts_nnet import Mcts
from gym_sokoban.envs import MctsSokobanEnv
from tests.testing_environment import unittest


RANDOM_SEED = 0
DIM_ROOM = (6, 6)
NUM_BOXES = 1
MAX_STEPS = 3
MAX_DEPTH = 10
MAX_ROLLOUTS = 10
SIMULATION_POLICY = "random"
NUM_PARALLEL = 8


class SetUpEnv(unittest.TestCase):
    """
    General class which setups the environment for all other tests.
    """

    def __init__(self, *args, **kwargs):
        super(SetUpEnv, self).__init__(*args, **kwargs)

    def setUp(self,
                 dim_room=DIM_ROOM,
                 num_boxes=NUM_BOXES,
                 max_steps=MAX_STEPS,
                 max_depth=MAX_DEPTH,
                 max_rollouts=MAX_ROLLOUTS,
                 simulation_policy=SIMULATION_POLICY,
                 num_parallel=NUM_PARALLEL,
                 random_seed=RANDOM_SEED,
                 print_board=False,
                 render_board=False):

        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.render_board = render_board

        self.mock_env = MctsSokobanEnv(
                dim_room=dim_room,
                max_steps=max_steps,
                num_boxes=num_boxes
        )

        self.mcts = Mcts(
                Env=self.mock_env,
                simulation_policy=simulation_policy,
                max_rollouts=max_rollouts,
                max_depth=max_depth,
                num_parallel=num_parallel
        )

        self.mock_env.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.mock_env.action_space.seed(random_seed)
        self.mock_env.reset()

        self.render_board = render_board
        self.print_board = print_board

        if self.print_board:
            print(f"\n---\nroom of size {dim_room} with {num_boxes} boxes and random_seed={random_seed}")
            print(self.mock_env.get_room_state())

        if self.render_board:
            self.mock_env.render_colored()
