from typing import Self

import torch
from torch import Tensor

from ._dqn_base_agent import DqnBaseAgent


class VanillaDQNAgent(DqnBaseAgent):
    """A vanilla deep-Q-network agent."""

    @classmethod
    @property
    def name(cls) -> str:
        return "vanilla_dqn"

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        return self.forward(next_states).max(1)[0].unsqueeze(1)
