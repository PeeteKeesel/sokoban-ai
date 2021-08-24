import numpy as np

from tests.set_up_env import SetUpEnv
from utils import manhattan_distance


RANDOM_SEED = 0
DIM_ROOM = (6, 6)
NUM_BOXES = 1
MAX_STEPS = 3
MAX_DEPTH = 10
MAX_ROLLOUTS = 10
SIMULATION_POLICY = "random"
NUM_PARALLEL = 8
INITIAL_ROOM_STATE = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 0],
                               [0, 0, 2, 1, 1, 0],
                               [0, 0, 0, 4, 1, 0],
                               [0, 0, 0, 5, 1, 0],
                               [0, 0, 0, 0, 0, 0]])


class TestSokobanEnv(SetUpEnv):
    """
    Tests functionality of SokobanEnv and MctsSokobanEnv.
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

    def test_state_after_action(self):
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)

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
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)

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
        self.assertEqual(
            2,
            len(children_not_None),
            "length of elements which are not None should be 2."
        )
        self.assertTrue(np.array_equal(children_expected, children_not_None))

    def test_manhattan_distance(self):
        self.setUp(dim_room=(6, 6), num_boxes=3, render_board=False)

        manh_dist_expect = 4
        manh_dist_actual = manhattan_distance(np.array([2, 3]), (5, 4))

        self.assertEqual(
            manh_dist_expect,
            manh_dist_actual,
            f"Manhattan distance between [2, 3] and [5, 4] should be {manh_dist_expect} but is {manh_dist_actual}"
        )

        self.assertRaises(AssertionError,
                          manhattan_distance, np.array([2, 3, 4]), (5, 4))

    def test_manhattan_heuristic(self):
        self.setUp(dim_room=(6, 6), num_boxes=3, render_board=False)


        manh_heur_expect = 1 + (1 + 1 + 2)
        manh_heur_actual = self.mock_env.manhattan_heuristic()

        self.assertEqual(
            manh_heur_expect,
            manh_heur_actual,
            f"Manhattan heuristic should be {manh_heur_expect} but is {manh_heur_actual} for \n{self.mock_env.room_state}"
        )

        self.setUp(dim_room=(7, 7), num_boxes=2, random_seed=10, render_board=True)
        self.mock_env.steps([2, 2, 2, 8, 6, 3])
        self.mock_env.render_colored()
        t = self.mock_env.manhattan_heuristic()
        print(t)

    def test_in_corner(self):
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)

        self.assertFalse(self.mock_env._in_corner())

        # 1st example
        self.mock_env.steps([8, 5, 5, 7, 2])
        self.assertTrue(self.mock_env._in_corner())

        # 2nd example
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)
        self.mock_env.steps([1, 1, 7, 5, 4])
        self.assertTrue(self.mock_env._in_corner())

        # 3rd example
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)
        self.mock_env.steps([1, 1, 8, 5, 3, 3])
        self.assertTrue(self.mock_env._in_corner())

        # 4th example: 3 walls around the box.
        self.setUp(dim_room=(8, 8), num_boxes=2, render_board=False)

        self.mock_env.steps([2, 8, 6, 6, 7, 3, 5, 4, 4])
        self.assertTrue(self.mock_env._in_corner())

    def test_deadlock_detection(self):
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)

        # Test corner deadlock.
        self.mock_env.steps([8, 5, 5, 7])
        self.assertTrue(self.mock_env.deadlock_detection(actionToTake=2),
                        f"The room state after taking action 2 should be a deadlock.")

        # Test corner deadlock with 3 walls around the box.
        self.setUp(dim_room=(8, 8), num_boxes=2, render_board=False)
        self.mock_env.steps([2, 8, 6, 6, 7, 3, 5, 4])
        self.assertTrue(self.mock_env.deadlock_detection(actionToTake=4),
                        f"The room state after taking action 2 should be a deadlock.")

        # TODO: test other deadlocks e.g. simple deadlocks


    ###########################################################################
    # static-methods                                                          #
    ###########################################################################
    def test_get_actions_lookup_chars(self):
        self.assertEqual(
            self.mock_env.get_actions_lookup_chars([2, 3, 4, 5]),
            ["D", "L", "R", "u"],
            "The chars for actions [2,3,4,5] should be ['D', 'L', 'R', 'u]''"
        )

        self.assertRaises(
            AssertionError,
            self.mock_env.get_actions_lookup_chars,
            [2, 3, 9, 5]
        )

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

    def test_get_best_immediate_action(self):
        self.setUp(dim_room=(6, 6), num_boxes=1, render_board=False)

        self.mock_env.steps([1, 8, 5])
        feasible_actions = self.mock_env.get_non_deadlock_feasible_actions()
        best_action = self.mock_env.get_best_immediate_action(feasible_actions)
        
        self.assertEqual(best_action, 3, "Best action should be 3 (L).")
