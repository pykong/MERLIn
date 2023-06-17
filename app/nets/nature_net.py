import torch
from torch import nn

from ._base_net import BaseNet


class NatureNet(BaseNet):
    @classmethod
    @property
    def name(cls) -> str:
        return "nature_net"

    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape  # unpack dimensions

        h_out_1 = self._calc_conv_outdim(x_dim, 8, 4, 1)
        w_out_1 = self._calc_conv_outdim(y_dim, 8, 4, 1)
        h_out_2 = self._calc_conv_outdim(h_out_1, 4, 2, 1)
        w_out_2 = self._calc_conv_outdim(w_out_1, 4, 2, 1)
        h_out_3 = self._calc_conv_outdim(h_out_2, 3, 1, 1)
        w_out_3 = self._calc_conv_outdim(w_out_2, 3, 1, 1)
        num_flat_features = h_out_3 * w_out_3

        # adapted from: https://github.com/KaleabTessera/DQN-Atari#dqn-neurips-architecture-implementation
        return nn.Sequential(
            # conv1
            nn.Conv2d(channel_dim, 32, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # conv3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # fc 1
            nn.Flatten(),
            nn.Linear(64 * num_flat_features, 256),
            nn.ReLU(),
            # output layer
            nn.Linear(256, num_actions),
        )
