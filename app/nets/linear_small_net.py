from torch import nn

from ._base_net import BaseNet


class LinearSmallNet(BaseNet):
    @classmethod
    @property
    def name(cls) -> str:
        return "linear_small_net"

    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape
        input_dims = channel_dim * x_dim * y_dim
        return nn.Sequential(
            # fc 1
            nn.Flatten(),
            nn.Linear(input_dims, 512),
            nn.ReLU(),
            # fc 2
            nn.Linear(512, 128),
            nn.ReLU(),
            # fc 3
            nn.Linear(128, 32),
            nn.ReLU(),
            # output
            nn.Linear(32, num_actions),
        )
