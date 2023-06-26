from typing import Final, Self

import torch
from torch import Tensor

from ._base_agent import BaseAgent


class DuellingDQNAgent(BaseAgent):
    """A duelling deep-Q-network agent."""

    name: Final[str] = "duelling_dqn"

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        q_values = self.forward(next_states)
        return q_values.max(dim=1)[0].unsqueeze(1).detach()

    def forward(self: Self, x):
        value, advantage = torch.split(self.model(x), [1, self.num_actions - 1], dim=1)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
