import random
from typing import Final, Self

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from torch import nn
from utils.logging import LogLevel, logger
from utils.replay_memory import ReplayMemory, Transition


class DuellingDQNAgent(BaseAgent):
    """An duelling deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "duelling_dqn"

    def _calc_max_q_prime(self: Self, next_states) -> float:
        value, advantage = torch.split(
            self.forward(next_states), [1, self.num_actions - 1], dim=1
        )
        max_q_prime = value + advantage - advantage.mean(dim=1, keepdim=True)
        return max_q_prime.max(dim=1)[0].unsqueeze(1).detach()
