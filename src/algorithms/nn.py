
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gym_sokoban.envs import SokobanEnv

NUM_FEATURES = 1024

class SokobanNN(nn.Module):
    """
    Neural network which predicts the probability distribution of actions
    and the value of the current state.
    """

    def __init__(self, Env: SokobanEnv, args):
        super(SokobanNN, self).__init__()

        self.board_x, self.board_y = Env.dim_room
        self.action_space = 8  #  TODO: get this from Env
        self.args = args

        self.conv1 = nn.Conv2d(1, args.num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1)

        self.batchn1 = nn.BatchNorm2d(args.num_channels)
        self.batchn2 = nn.BatchNorm2d(args.num_channels)
        self.batchn3 = nn.BatchNorm2d(args.num_channels)
        self.batchn4 = nn.BatchNorm2d(args.num_channels)

        self.fully_connected1 = nn.Linear(args.num_channels * (self.board_x-...) * (self.board_y-...), NUM_FEATURES)
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

