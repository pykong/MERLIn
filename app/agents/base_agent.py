import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from utils.logging import LogLevel, logger
from utils.replay_memory import ReplayMemory, Transition


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


class BaseAgent(ABC, pl.LightningModule):
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
        weight_decay=1e-5,
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
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=alpha, weight_decay=weight_decay
        )

    @abstractmethod
    def _make_model(
        self, state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        pass

    @abstractmethod
    def replay(self: Self) -> None:
        pass

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    def _prepare_minibatch(self: Self, minibatch: list[Transition]):
        # states, actions, rewards, next_states, dones = zip(*minibatch)
        # states = torch.from_numpy(np.array(states)).float().to(self.device_)
        # actions = torch.tensor(actions).unsqueeze(1).to(self.device_)
        # rewards = torch.tensor(rewards).float().to(self.device_)
        # next_states = torch.from_numpy(np.array(next_states)).float().to(self.device_)
        # dones = torch.tensor(dones).float().to(self.device_)
        # return states, actions, rewards, next_states, dones

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in minibatch:
            states.append(transition.state)
            actions.append([transition.action])
            rewards.append([transition.reward])
            next_states.append(transition.next_state)
            dones.append([transition.done])

        states = torch.from_numpy(np.array(states)).float().to(self.device_)
        actions = torch.tensor(actions).to(self.device_)
        rewards = torch.tensor(rewards).to(self.device_)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device_)
        dones = torch.tensor(dones).float().to(self.device_)

        return states, actions, rewards, next_states, dones

    def _update_weights(self, q_a, target) -> None:
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_a, target)
        loss.backward()
        self.optimizer.step()

    def remember(self: Self, transition: Transition) -> None:
        self.memory.push(transition)

    def act(self: Self, state) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device_)
        act_values = self.forward(state)
        return act_values.argmax().item()

    def forward(self: Self, x):
        return self.model(x)

    def update_epsilon(self: Self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.gamma

    def load(self: Self, name: Path) -> None:
        self.load_state_dict(torch.load(name))

    def save(self: Self, name: Path) -> None:
        torch.save(self.state_dict(), name)
