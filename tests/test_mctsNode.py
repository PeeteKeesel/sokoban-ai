import random
import time
import numpy as np

from gym_sokoban.envs.sokoban_env  import SokobanEnv, ACTION_LOOKUP_CHARS
from tests.testing_environment     import unittest
from src.algorithms.mcts           import MctsNode

RANDOM_SEED = 0
INITIAL_ROOM_6x6_1 = np.array([[0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 0],
                                [0, 0, 2, 1, 1, 0],
                                [0, 0, 0, 4, 1, 0],
                                [0, 0, 0, 5, 1, 0],
                                [0, 0, 0, 0, 0, 0]])

# ================================================================
class TestMctsNode(unittest.TestCase):

    def setUp(self, dim_room=(6, 6), num_boxes=1, render_board=False,
              print_board=False, random_seed=RANDOM_SEED):
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

    def test_constructor(self):
        print("test_constructor()")
        self.setUp(print_board=False, render_board=True)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())

        self.assertEqual(0, root.depth)
        self.assertTrue(np.alltrue(INITIAL_ROOM_6x6_1 == root.room_state))
        self.assertEqual(9, root.n_actions)
        self.assertIsNone(root.prev_action)
        self.assertFalse(root.is_expanded)
        self.assertEqual(0, root.n_vlosses)
        self.assertTrue(9 == len(root.child_N) == len(root.child_W) ==
                        len(root.original_P) == len(root.child_P))
        self.assertEqual({}, root.children)

    def test_N_setter(self):
        print("test_N_setter()")
        self.setUp(print_board=False, render_board=True)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())

        root.prev_action = 2
        root.N = 5
        self.assertEqual(5, root.N)
        self.assertEqual(5, root.parent.child_N[2])
        self.assertEqual(0, root.parent.child_N[1])

    def test_maybe_add_child(self):
        print("test_maybe_add_child()")
        self.setUp(render_board=True)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())

        self.assertTrue(1 not in root.children)
        self.assertTrue(2 not in root.children)

        child_after_1 = root.maybe_add_child(1)  # Feasible action
        child_after_2 = root.maybe_add_child(2)  # Non feasible action

        self.assertTrue(1 in root.children)
        self.assertTrue(2 in root.children)
        self.assertFalse(3 in root.children)

        room_after_action_1 = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 2, 4, 1, 0],
            [0, 0, 0, 5, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]])

        print(child_after_1.room_state)

        self.assertTrue(np.alltrue(room_after_action_1 == root.children[1].room_state),
                        "The room_state of the child after taking action 1 is wrong.")
        self.assertTrue(np.alltrue(INITIAL_ROOM_6x6_1   == root.children[2].room_state),
                        "The room_state of the child after taking action 2 should not change.")

        self.assertTrue(np.alltrue(root.children[1].room_state == child_after_1.room_state))
        self.assertTrue(np.alltrue(root.children[2].room_state == child_after_2.room_state))

        self.assertEqual(root, root.children[1].parent,
                         "The parent of the child after action 1 should be the root.")
        self.assertEqual(root, root.children[2].parent,
                         "The parent of the child after action 2 should be the root.")

        # Try to add an already existing child node
        root.maybe_add_child(2)

    def test_select_until_leaf(self):
        print("test_select_until_leaf()")
        self.setUp(render_board=True)

        pass

    def test_print_tree(self):
        print("test_print_tree()")
        self.setUp(render_board=False)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())
        root.maybe_add_child(1)  # Feasible action
        root.maybe_add_child(2)  # Non feasible action
        root.maybe_add_child(3)  # Non feasible action
        root.maybe_add_child(4)  # Non feasible action
        root.maybe_add_child(5)  # Non feasible action
        root.maybe_add_child(6)  # Non feasible action
        root.maybe_add_child(7)  # Non feasible action
        root.maybe_add_child(8)  # Feasible action

        root.print_tree()
