import random
from copy import deepcopy
from pathlib import Path
from typing import Final, Self

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.base_agent import BaseAgent
from torch import nn
from utils.logging import LogLevel, logger
from utils.replay_memory import ReplayMemory, Transition


class DDQNCNNAgent(BaseAgent):
    """An double deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "double_dqn_cnn"

    def __init__(
        self: Self,
        *args,
        **kwargs,
    ):
        self.target_net_update_interval = kwargs.pop("target_net_update_interval")
        self._step_counter: int = 0
        super().__init__(*args, **kwargs)
        self.target_model = deepcopy(self.model)

    @staticmethod
    def _make_model(
        state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape  # unpack dimensions
        model = nn.Sequential(
            # 1 - conv
            nn.Conv2d(channel_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # 2 - conv
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # # 3 - conv
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            # 4 - fc
            nn.Flatten(),
            nn.Linear(32 * (x_dim // 4) * (y_dim // 4), 64),
            nn.ReLU(),
            # 5 - fc
            nn.Linear(64, 8),
            nn.ReLU(),
            # 6 - fc
            # nn.Linear(32, 8),
            # nn.ReLU(),
            # 7 - fc (output layer)
            nn.Linear(8, num_actions),
        )
        model.to(device)
        return model

    def replay(self: Self) -> None:
        # sample memory
        minibatch = self.memory.sample(self.batch_size)

        # convert the minibatch to a more convenient format
        states, actions, rewards, next_states, dones = self._prepare_minibatch(
            minibatch
        )

        # predict Q-values for the initial states.
        q_out = self.forward(states)
        q_a = q_out.gather(1, actions)  # state_action_values

        # get indices of maximum values according to the policy network
        _, policy_net_actions = self.forward(next_states).max(1)

        # compute V(s_{t+1}) for all next states using target network, but choose the best action from the policy network.
        max_q_prime = (
            self.target_model(next_states)
            .gather(1, policy_net_actions.unsqueeze(-1))
            .squeeze()
            .detach()
        )

        # mask dones
        dones = 1 - dones

        # compute the expected Q values (expected_state_action_values)
        target = rewards + max_q_prime * self.gamma * dones

        # scale target
        # target /= 100

        # clip target
        # target = target.clamp(min=-1.0, max=1.0)

        # update the weights.
        self._update_weights(q_a, target.unsqueeze(1))

        # target update logic
        self._step_counter += 1
        if self._step_counter % self.target_net_update_interval == 0:
            self.__update_target()

    def __update_target(self: Self) -> None:
        """Copies the policy network parameters to the target network"""
        self.target_model.load_state_dict(self.model.state_dict())
