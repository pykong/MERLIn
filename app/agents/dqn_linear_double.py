from copy import deepcopy
from typing import Final, Self

import numpy as np
import torch
from agents.base_agent import BaseAgent
from torch import nn


class DDQNLinearAgent(BaseAgent):
    """An double deep-Q-network agent with a convolutional neural network structure."""

    def __init__(
        self: Self,
        *args,
        **kwargs,
    ):
        self.target_net_update_interval = kwargs.pop("target_net_update_interval")
        self.epochs = kwargs.pop("epochs")
        self._step_counter: int = 0
        super().__init__(*args, **kwargs)
        self.target_model = deepcopy(self.model)

    @classmethod
    @property
    def name(cls) -> str:
        return "double_linear_dqn"

    @staticmethod
    def _make_model(
        state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        # channel_dim, x_dim, y_dim = state_shape  # unpack dimensions
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_actions),
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

        # mask dones
        dones = 1 - dones
        for _ in range(self.epochs):
            # predict Q-values for the initial states.
            q_out = self.forward(states)
            q_a = q_out.gather(1, actions)  # state_action_values

            max_q_prime = self.target_model(next_states).max(1)[0].unsqueeze(1)

            # scale rewards
            rewards /= 100

            # compute the expected Q values (expected_state_action_values)
            target = rewards + self.gamma * max_q_prime * dones

            # update the weights.
            self._update_weights(q_a, target)

        # target update logic
        self._step_counter += 1
        if self._step_counter % self.target_net_update_interval == 0:
            self.__update_target()

    def __update_target(self: Self) -> None:
        """Copies the policy network parameters to the target network"""
        self.target_model = deepcopy(self.model)
