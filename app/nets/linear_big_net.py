from torch import nn

from ._base_net import BaseNet


class LinearBigNet(BaseNet):
    @classmethod
    @property
    def name(cls) -> str:
        return "linear_big_net"

    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape
        input_dims = channel_dim * x_dim * y_dim
        return nn.Sequential(
            # fc 1
            nn.Flatten(),
            nn.Linear(input_dims, 1024),
            nn.ReLU(),
            # fc 2
            nn.Linear(1024, 512),
            nn.ReLU(),
            # fc 3
            nn.Linear(512, 128),
            nn.ReLU(),
            # fc 4
            nn.Linear(128, 32),
            nn.ReLU(),
            # fc 5
            nn.Linear(32, 8),
            nn.ReLU(),
            # output
            nn.Linear(8, num_actions),
        )
