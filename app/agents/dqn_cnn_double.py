import random
from copy import deepcopy
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
        self: Self,
        state_shape: tuple[int, int, int],
        action_space: int,
        alpha: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        gamma: float = 0.999,  # epsilon decay
        memory_size: int = 10_000,
        batch_size: int = 64,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = action_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.device_: torch.device = get_torch_device()
        self.model = self._make_model(self.state_shape, self.num_actions, self.device_)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.target_model = deepcopy(self.model)  # init target network

    def update_target(self: Self) -> None:
        """Copies the policy network parameters to the target network"""
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def _make_model(
        state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        channel_dim, x_dim, y_dim = state_shape  # unpack dimensions
        model = nn.Sequential(
            nn.Conv2d(channel_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * (x_dim // 2) * (y_dim // 2), 64),  # fully connected layer
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        model.to(device)
        return model

    def remember(self: Self, experience: Experience) -> None:
        self.memory.push(experience)

    def act(self: Self, state) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device_)
        act_values = self.forward(state)
        return int(torch.argmax(act_values[0]).item())

    def replay(self: Self) -> None:
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

    def __update_weights(self: Self, q_a, target) -> None:
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
