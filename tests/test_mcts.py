import unittest

from agents.algorithms.mcts import Mcts, MctsNode
from tests.testing_environment import *



# ================================================================
class TestMCTS(unittest.TestCase):

    def test_do(self):
        mcts = Mcts(MctsNode(ROOM_STATE_L1.reshape(-1)))


        # algos.do(ROOM_STATE_L1.reshape(-1), (4, 4))



