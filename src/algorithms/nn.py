import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gym_sokoban.envs import SokobanEnv

NUM_FEATURES = 1024

import argparse
description = "Sokoban AlphaGo"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-lr',   '--learning_rate', type=float, dest='learningRate', default=.001, help="Learning rate")
parser.add_argument('-dout', '--drop_out',      type=float, dest="dropout",      default=.3,   help="Drop out")
parser.add_argument('-ep',   '--epochs',        type=int,   dest="epochs",       default=10,   help="Epochs of training")
parser.add_argument('-bs',   '--batch_size',    type=int,   dest="batchSize",    default=64,   help="Batch size")
parser.add_argument('-nCh',  '--num_channels',  type=int,   dest="numChannels",  default=3,    help="Number of channels")
parser.add_argument('-nIter','--num_iters',     type=int,   dest="numIters",     default=1000, help="Number of training iterations")
parser.add_argument('-nEps', '--num_eps',       type=int,   dest="numEps",       default=100,  help="Number of episodes")
parser.add_argument('-nMctsSim', '--num_mcts_sim', type=int,dest="numMctsSim",   default=20,   help="Number of MCTS simulations")
args = parser.parse_args()

class SokobanNN(nn.Module):
    """
    Neural network which predicts the probability distribution of actions
    and the value of the current state.
    """

    def __init__(self, Env: SokobanEnv, args):
        super(SokobanNN, self).__init__()

        self.dim_x, self.dim_y = Env.dim_room
        self.action_space = len(SokobanEnv.get_action_lookup())
        self.args = args

        self.conv1 = nn.Conv2d(1, args.numChannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.numChannels, args.numChannels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.numChannels, args.numChannels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(args.numChannels, args.numChannels, kernel_size=3, stride=1)

        self.batchn1 = nn.BatchNorm2d(args.numChannels)
        self.batchn2 = nn.BatchNorm2d(args.numChannels)
        self.batchn3 = nn.BatchNorm2d(args.numChannels)
        self.batchn4 = nn.BatchNorm2d(args.numChannels)

        self.fully_connected1 = nn.Linear(args.numChannels *
                                            (self.dim_x-...) * (self.dim_y-...),
                                          NUM_FEATURES)
        self.fully_connected_batchn1 = nn.BatchNorm1d(NUM_FEATURES)

        self.fully_connected2 = nn.Linear(NUM_FEATURES, 512)
        self.fully_connected_batchn1 = nn.BatchNorm1d(512)

        self.fully_connected3 = nn.Linear(512, self.action_space)
        self.fully_connected4 = nn.Linear(512, 1)

    def forward(self, data):

        data_one_hot = torch.zeros((data.shape[0], self.n_obs))
        data_one_hot[np.arange(data.shape[0]), data] = 1.0

        s = F.relu(self.bn1(self.conv1(data_one_hot)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        h_relu = F.relu(self.dense1(data_one_hot))
        logits = self.dense_p(h_relu)
        policy = F.softmax(logits, dim=1)

        value = self.dense_v(h_relu).view(-1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs)
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()