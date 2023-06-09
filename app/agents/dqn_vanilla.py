from typing import Self

import torch
from agents.base_agent import BaseAgent
from torch import nn


class VanillaDQNAgent(BaseAgent):
    """An vanilla deep-Q-network agent with a convolutional neural network structure."""

    @classmethod
    @property
    def name(cls) -> str:
        return "vanilla_dqn"

    @staticmethod
    def _make_model(
        state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape  # unpack dimensions

        # calculate the size of the output of the last conv layer:
        def calc_dim(dim: int, kernel_size: int, stride: int, padding: int) -> int:
            return ((dim + 2 * padding - kernel_size) // stride) + 1

        h_out_1 = calc_dim(x_dim, 8, 4, 1)
        w_out_1 = calc_dim(y_dim, 8, 4, 1)
        h_out_2 = calc_dim(h_out_1, 4, 2, 1)
        w_out_2 = calc_dim(w_out_1, 4, 2, 1)
        num_flat_features = h_out_2 * w_out_2

        # adapted from: https://github.com/KaleabTessera/DQN-Atari#dqn-neurips-architecture-implementation
        model = nn.Sequential(
            # conv1
            nn.Conv2d(channel_dim, 16, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # conv2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # fc 1
            nn.Flatten(),
            nn.Linear(32 * num_flat_features, 256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # fc 2 - additional layer in contrast to NeuroIPS paper
            nn.Linear(256, 32),
            nn.ELU(),
            # fc 3 - additional layer in contrast to NeuroIPS paper
            nn.Linear(32, 16),
            nn.ReLU(),
            # output
            nn.Linear(16, num_actions),
        )
        model.to(device)
        return model

    def _calc_max_q_prime(self: Self, next_states) -> float:
        return self.model(next_states).max(1)[0].unsqueeze(1)
