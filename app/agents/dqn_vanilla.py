from typing import Self

import torch
from torch import Tensor

from ._base_agent import BaseAgent


class VanillaDQNAgent(BaseAgent):
    """An vanilla deep-Q-network agent with a convolutional neural network structure."""

    @classmethod
    @property
    def name(cls) -> str:
        return "vanilla_dqn"

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        return self.forward(next_states).max(1)[0].unsqueeze(1)
