import random
from pathlib import Path
from typing import Final, Self

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from utils.logging import LogLevel, logger
from utils.replay_memory import Experience, ReplayMemory


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        logger.log(str(LogLevel.GREEN), f"CUDA is available (v{torch.version.cuda}).")
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_name(i)
            logger.log(str(LogLevel.GREEN), f"cuda:{i} - {gpu}")
        return torch.device("cuda")
    else:
        logger.log(str(LogLevel.YELLOW), f"Running PyTorch on CPU.")
        return torch.device("cpu")


class DDQNCNNAgent(pl.LightningModule):
    """An double deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "double_dqn_cnn"

    def __init__(
        self,
        state_shape,
        action_space,
        alpha=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        gamma=0.999,
        memory_size=10_000,
        batch_size=64,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_space = action_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.model: nn.Sequential = self._build_model()
        self.device_: torch.device = get_torch_device()
        self.model.to(self.device_)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.target_model: nn.Sequential = self._build_model()  # Target network
        self.target_model.to(self.device_)  # TODO: Neccessary?
        self.update_target()  # Initialize target network weights to be the same as policy network

    def update_target(self):
        """Copies the policy network parameters to the target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    def _build_model(self) -> nn.Sequential:
        # Calculate the output size after the convolutional layers
        conv_output_size = self.state_shape[1] // 4  # two conv layers with stride=2
        conv_output_size *= self.state_shape[2] // 4  # two conv layers with stride=2
        conv_output_size *= 16  # output channels of last conv layer
        return nn.Sequential(
            nn.Conv2d(self.state_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_space),
        )

    def remember(self, experience: Experience) -> None:
        self.memory.push(experience)

    def act(self: Self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device_)
        act_values = self.forward(state)
        return int(torch.argmax(act_values[0]).item())

    def replay(self) -> None:
        # sample memory
        minibatch = self.memory.sample(self.batch_size)

        # convert the minibatch to a more convenient format
        states, actions, rewards, next_states, dones = self.prepare_minibatch(minibatch)

        # predict Q-values for the initial states.
        q_out = self.forward(states)
        q_a = q_out.gather(1, actions)  # state_action_values

        # compute V(s_{t+1}) for all next states.
        max_q_prime = self.target_model.forward(next_states).max(1)[0].detach()

        # scale rewards
        rewards /= 100

        # clip rewards
        rewards = rewards.clamp(min=-1.0, max=1.0)

        # mask dones
        dones = 1 - dones

        # compute the expected Q values (expected_state_action_values)
        target = rewards + max_q_prime * self.gamma * dones

        # update the weights.
        self.__update_weights(q_a, target.unsqueeze(1))

    def __update_weights(self, q_a, target) -> None:
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_a, target)
        loss.backward()
        self.optimizer.step()

    def prepare_minibatch(self: Self, minibatch: list[Experience]):
        # TODO: make private
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.from_numpy(np.array(states)).float().to(self.device_)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device_)
        rewards = torch.tensor(rewards).float().to(self.device_)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device_)
        dones = torch.tensor(dones).float().to(self.device_)
        return states, actions, rewards, next_states, dones

    def forward(self: Self, x):
        return self.model(x)

    def update_epsilon(self: Self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.gamma

    def load(self: Self, name: Path) -> None:
        self.load_state_dict(torch.load(name))

    def save(self: Self, name: Path) -> None:
        torch.save(self.state_dict(), name)
