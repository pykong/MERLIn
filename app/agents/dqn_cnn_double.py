from copy import deepcopy
from typing import Final, Self

import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from torch import nn


class DDQNCNNAgent(BaseAgent):
    """An double deep-Q-network agent with a convolutional neural network structure."""

    def __init__(
        self: Self,
        epochs: int = 1,
        *args,
        **kwargs,
    ):
        self.epochs = epochs
        self.target_net_update_interval = kwargs.pop("target_net_update_interval")
        self._step_counter: int = 0
        super().__init__(*args, **kwargs)
        self.target_model = deepcopy(self.model)

    @classmethod
    @property
    def name(cls) -> str:
        return "double_dqn_cnn"

    @staticmethod
    def _make_model(
        state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape  # unpack dimensions
        # adapted from: https://github.com/KaleabTessera/DQN-Atari#dqn-neurips-architecture-implementation
        model = nn.Sequential(
            # conv1
            nn.Conv2d(channel_dim, 16, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # conv2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # fc 1
            nn.Flatten(),
            # nn.Linear(32 * (x_dim // 16) * (y_dim // 16), 256),
            nn.Linear(32 * 410, 256),  # TODO Dynamically correct input size
            nn.ReLU(),
            nn.Dropout(0.2),
            # fc 2 - additional layer in contrast to NeuroIPS paper
            nn.Linear(256, 32),
            nn.ReLU(),
            # output
            nn.Linear(32, num_actions),
        )
        model.to(device)
        return model

    def replay(self: Self) -> float:
        # sample memory
        minibatch = self.memory.sample(self.batch_size)

        # convert the minibatch to a more convenient format
        states, actions, rewards, next_states, dones = self._prepare_minibatch(
            minibatch
        )

        # predict Q-values for the initial states.
        # q_out = self.forward(states)
        # q_a = q_out.gather(1, actions)  # state_action_values

        # get indices of maximum values according to the policy network
        # _, policy_net_actions = self.forward(next_states).max(1)

        # compute V(s_{t+1}) for all next states using target network, but choose the best action from the policy network.
        # max_q_prime = (
        #     self.target_model(next_states)
        #     .gather(1, policy_net_actions.unsqueeze(-1))
        #     .squeeze()
        #     .detach()
        # )

        # mask dones
        dones = 1 - dones
        for _ in range(self.epochs):
            # predict Q-values for the initial states.
            q_out = self.forward(states)
            q_a = q_out.gather(1, actions)  # state_action_values

            with torch.no_grad():
                max_q_prime = self.target_model(next_states).max(1)[0].unsqueeze(1)

            # compute the expected Q values (expected_state_action_values)
            target = rewards + self.gamma * max_q_prime * dones

            losses = F.smooth_l1_loss(q_a, target)

            # update the weights.
            self._update_weights(losses)

        # target update logic
        self._step_counter += 1
        if self._step_counter % self.target_net_update_interval == 0:
            self.__update_target()

        return losses.mean().item()

    def __update_target(self: Self) -> None:
        """Copies the policy network parameters to the target network"""
        self.target_model = deepcopy(self.model)
