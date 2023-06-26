from copy import deepcopy
from typing import Self

import torch
from torch import Tensor

from ._base_agent import BaseAgent


class DoubleDQNAgent(BaseAgent):
    """A double deep-Q-network agent."""

    def __init__(
        self: Self,
        *args,
        **kwargs,
    ):
        self.target_net_update_interval = kwargs.pop("target_net_update_interval")
        self._step_counter: int = 0
        super().__init__(*args, **kwargs)
        self.target_model = deepcopy(self.model)

    @classmethod
    @property
    def name(cls) -> str:
        return "double_dqn"

    def replay(self: Self) -> float:
        # target update logic
        self._step_counter += 1
        if self._step_counter % self.target_net_update_interval == 0:
            self.target_model = deepcopy(self.model)
        return super().replay()

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        return self.target_model(next_states).max(1)[0].unsqueeze(1)
