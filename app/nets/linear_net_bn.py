from torch import nn

from ._base_net import BaseNet


class LinearNetBN(BaseNet):
    """Linear net with batch normalization."""

    @classmethod
    @property
    def name(cls) -> str:
        return "linear_net_bn"

    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape
        input_dims = channel_dim * x_dim * y_dim
        return nn.Sequential(
            # fc 1
            nn.Flatten(),
            nn.Linear(input_dims, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # fc 2
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # fc 3
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # fc 4
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # output
            nn.Linear(16, num_actions),
        )