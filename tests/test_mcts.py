import random
import time
import numpy as np

from gym_sokoban.envs.sokoban_env  import SokobanEnv, ACTION_LOOKUP_CHARS
from tests.testing_environment     import unittest
from src.algorithms.mcts import Mcts, MctsNode, execute_episode

RANDOM_SEED = 0
INITIAL_ROOM_6x6_1 = np.array([[0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 0],
                                [0, 0, 2, 1, 1, 0],
                                [0, 0, 0, 4, 1, 0],
                                [0, 0, 0, 5, 1, 0],
                                [0, 0, 0, 0, 0, 0]])

# ================================================================
class TestMcts(unittest.TestCase):

    def setUp(self, dim_room=(6, 6), num_boxes=1, max_steps=150,
              render_board=False, print_board=False, random_seed=RANDOM_SEED):
        self.mock_env = SokobanEnv(dim_room=dim_room,
                                   max_steps=max_steps,
                                   num_boxes=num_boxes)

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

    def test_sp_uct(self):
        # TODO: optional to implement this unit test
        self.setUp()
        np.random.seed(0)

        mcts = Mcts(self.mock_env)
        mcts.initialize_search()
        # set some mock values for N and W
        mcts.root.child_N = np.arange(0,9) # np.random.randint(low=1, high=2, size=9)
        mcts.root.child_W = np.random.randint(5, size=9)

        i_child = None
        # add mock children nodes for all actions
        for i in range(1, 9):
            #mcts.root.inject_noise()
            i_child = mcts.root.maybe_add_child(i)


        for j in range(2, 9):
            i_child.inject_noise()
            i_child.maybe_add_child(j)

        i_child.child_N = np.arange(9, 18)  # np.random.randint(low=1, high=2, size=9)
        i_child.child_W = np.random.randint(5, size=9)

        mcts.root.select_until_leaf()
        mcts.root.print_tree()

        print("..................")
        print(i_child.sp_uct)

        self.assertEqual(1, 1, "...")

        pass

    def test_execute_episode(self):
        print("test_execute_episode()")
        self.setUp(dim_room=(7,7), max_steps=10, render_board=True, num_boxes=2)

        # num_episodes = 10
        # for i in range(num_episodes):
        #     print(200*"$$"+" episode " + str(i))
        execute_episode(numSimulations=50, Env=self.mock_env)



        pass