import random
from typing import Final

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


class DQNCNNAgent(pl.LightningModule):
    """An deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "dqn_cnn"

    def __init__(
        self,
        state_shape,
        action_space,
        alpha=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        memory_size=10_000,
        batch_size=64,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = epsilon_decay
        self.memory = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.model: nn.Sequential = self._build_model()
        self.gpu = get_torch_device()
        self.model.to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

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
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
        )

    def remember(self, experience: Experience) -> None:
        self.memory.push(experience)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.gpu)
        act_values = self.model(state)
        return int(torch.argmax(act_values[0]).item())

    def replay(self) -> None:
        minibatch = self.memory.sample(self.batch_size)

        # Convert the minibatch to a more convenient format.
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and add an extra dimension.
        states = torch.from_numpy(np.array(states)).float().to(self.gpu)
        actions = torch.tensor(actions).unsqueeze(1).to(self.gpu)
        rewards = torch.tensor(rewards).float().to(self.gpu)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.gpu)
        dones = torch.tensor(dones).float().to(self.gpu)

        # Predict Q-values for the initial states.
        q_out = self.forward(states)
        q_a = q_out.gather(1, actions)  # state_action_values

        # Compute V(s_{t+1}) for all next states.
        max_q_prime = self.model(next_states).max(1)[0].detach()  # next_state_values

        # Compute the expected Q values (expected_state_action_values)
        target = (max_q_prime * self.gamma + rewards) * (1 - dones)

        # Update the weights.
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_a, target.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        return self.model(x)

    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.gamma

    def load(self, name) -> None:
        self.load_state_dict(torch.load(name))

    def save(self, name) -> None:
        torch.save(self.state_dict(), name)
