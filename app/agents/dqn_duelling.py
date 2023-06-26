from typing import Final, Self

import torch
from torch import Tensor

from ._base_agent import BaseAgent


class DuellingDQNAgent(BaseAgent):
    """An duelling deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "duelling_dqn"

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        q_prime = self.forward(next_states)  # TODO: Is q_prime the right variable name?
        value, advantage = torch.split(q_prime, [1, self.num_actions - 1], dim=1)
        max_q_prime = value + advantage - advantage.mean(dim=1, keepdim=True)
        return max_q_prime.max(dim=1)[0].unsqueeze(1).detach()
