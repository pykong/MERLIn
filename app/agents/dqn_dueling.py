from typing import Self

import torch
from torch import Tensor

from ._base_agent import BaseAgent


class DuelingDQNAgent(BaseAgent):
    """A dueling deep-Q-network agent."""

    @classmethod
    @property
    def name(cls) -> str:
        return "dueling_dqn"

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        raise NotImplementedError()
