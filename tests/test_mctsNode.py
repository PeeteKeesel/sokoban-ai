import numpy as np

from tests.set_up_env import SetUpEnv
from src.algorithms.mcts import MctsNode
from src.utils import ACTION_LOOKUP


RANDOM_SEED = 0
DIM_ROOM = (6, 6)
NUM_BOXES = 1
MAX_STEPS = 3
MAX_DEPTH = 10
MAX_ROLLOUTS = 10
SIMULATION_POLICY = "random"
NUM_PARALLEL = 8
INITIAL_ROOM_6x6_1 = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 0, 2, 1, 1, 0],
                               [0, 0, 0, 4, 1, 0],
                               [0, 0, 0, 5, 1, 0],
                               [0, 0, 0, 0, 0, 0]])


class TestMctsNode(SetUpEnv):
    """
    Tests functionality of MctsNode.
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

    def test_constructor(self):
        print("test_constructor()")
        self.setUp(print_board=False, render_board=True)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())

        self.assertEqual(0, root.depth)
        self.assertTrue(np.alltrue(INITIAL_ROOM_6x6_1 == root.room_state))
        self.assertEqual(len(ACTION_LOOKUP.keys()), root.n_actions)
        self.assertIsNone(root.prev_action)
        self.assertFalse(root.is_expanded)
        self.assertEqual(0, root.n_vlosses)
        self.assertTrue(len(ACTION_LOOKUP.keys()) == len(root.child_N) == len(root.child_W) ==
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
        self.setUp(render_board=False)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())
        child_after_1 = root.maybe_add_child(1)
        root.maybe_add_child(2)
        root.maybe_add_child(3)
        child_after_8 = root.maybe_add_child(4)

        root.child_P = np.array(
            [0.00, 0.13, 0.01, 0.01, 0.02],
            dtype=np.float32)

        # Mock as if all nodes were expanded.
        root.is_expanded = True

        # Go the trajectory [U, R, U, L] and [U, U]
        child_after_1.maybe_add_child(1)
        child_after_1.is_expanded = True
        child_after_1.child_P = np.array(
            [0.00, 0.2, 0.01, 0.01, 0.32],
            dtype=np.float32)
        child_after_14 = child_after_1.maybe_add_child(4)

        child_after_141 = child_after_14.maybe_add_child(1)
        child_after_14.is_expanded = True
        child_after_14.child_P = np.array(
            [0.00, 0.51, 0.01, 0.01, 0.02],
            dtype=np.float32)

        # Should be a terminal state.
        child_after_141.maybe_add_child(3)
        child_after_141.is_expanded = True
        child_after_141.child_P = np.array(
            [0.00, 0.61, 0.01, 0.61, 0.02],
            dtype=np.float32)

        # Go the trajectory [r, u] and [r, D]
        child_after_8.maybe_add_child(2)
        child_after_8.is_expanded = True
        child_after_8.maybe_add_child(1)

        leaf = root.select_until_leaf()

        if self.render_board:
            root.print_tree()
            leaf.Env.render_colored()

    def test_print_tree(self):
        print("test_print_tree()")
        self.setUp(render_board=True)

        root = MctsNode(self.mock_env, self.mock_env.get_n_actions())
        root.maybe_add_child(1)  # Feasible action
        root.maybe_add_child(2)  # Non feasible action
        root.maybe_add_child(3)  # Non feasible action
        child_after_8 = root.maybe_add_child(4)  # Feasible action

        child_after_8.maybe_add_child(3)

        if self.render_board:
            root.print_tree()

    def test_execute_episode(self):
        print("test_execute_episode()")

        # execute_episode_with_nnet(numSimulatons=2, env=SokobanEnv)

        pass

    def test_maybe_add_child_transposition_tables(self):
        self.setUp(render_board=True)

        pass