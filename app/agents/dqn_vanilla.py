from typing import Self

from ._base_agent import BaseAgent


class VanillaDQNAgent(BaseAgent):
    """An vanilla deep-Q-network agent with a convolutional neural network structure."""

    @classmethod
    @property
    def name(cls) -> str:
        return "vanilla_dqn"

    def _calc_max_q_prime(self: Self, next_states) -> float:
        return self.model(next_states).max(1)[0].unsqueeze(1)
