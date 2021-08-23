import numpy as np

from tests.setUpEnv import SetUpEnv
from src.algorithms.mcts import execute_episode


RANDOM_SEED = 0
DIM_ROOM = (6, 6)
NUM_BOXES = 1
MAX_STEPS = 3
MAX_DEPTH = 10
MAX_ROLLOUTS = 10
SIMULATION_POLICY = "random"
NUM_PARALLEL = 8


class TestMcts(SetUpEnv):
    """
    Tests functionality of Mcts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setUp(
            dim_room=DIM_ROOM,
            num_boxes=NUM_BOXES,
            max_steps=MAX_STEPS,
            max_depth=MAX_DEPTH,
            max_rollouts=MAX_ROLLOUTS,
            simulation_policy=SIMULATION_POLICY,
            num_parallel=NUM_PARALLEL,
            random_seed=RANDOM_SEED,
            render_board=False,
            print_board=False
        )

    def test_sp_uct(self):
        self.setUp()
        np.random.seed(0)

        self.mcts.initialize_search()
        # set some mock values for N and W
        self.mcts.root.child_N = np.arange(0,9) # np.random.randint(low=1, high=2, size=9)
        self.mcts.root.child_W = np.random.randint(5, size=9)

        i_child = None
        # add mock children nodes for all actions
        for i in range(1, 9):
            #mcts.root.inject_noise()
            i_child = self.mcts.root.maybe_add_child(i)


        for j in range(2, 9):
            i_child.inject_noise()
            i_child.maybe_add_child(j)

        i_child.child_N = np.arange(9, 18)  # np.random.randint(low=1, high=2, size=9)
        i_child.child_W = np.random.randint(5, size=9)

        self.mcts.root.select_until_leaf()
        self.mcts.root.print_tree()

        print("..................")
        print(i_child.sp_uct)

        self.assertEqual(1, 1, "...")

        pass

    def test_execute_episode(self):
        print("test_execute_episode()")
        self.setUp(dim_room=(6, 6), max_steps=5, render_board=True, num_boxes=1)

        # num_episodes = 10
        # for i in range(num_episodes):
        #     print(200*"$$"+" episode " + str(i))
        execute_episode(numSimulations=5,  # number of simulations per state.
                        Env=self.mock_env,
                        max_rollouts=10,   # number of times an action will be picked
                                           # after the simulations.
                        max_depth=6)      # max number of steps per simulation.

        pass