import random
from typing import Final

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
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


class DuellingQNCNNAgent(pl.LightningModule):
    """An duelling deep-Q-network agent with a convolutional neural network structure."""

    name: Final[str] = "duelling_dqn_cnn"

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
        # TODO: rename model -> feature
        self.model, self.advantage, self.value = self._build_model()
        self.gpu = get_torch_device()
        self.model.to(self.gpu)
        self.advantage.to(self.gpu)
        self.value.to(self.gpu)

    def _build_model(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        # Calculate the output size after the convolutional layers
        conv_output_size = self.state_shape[1] // 4  # two conv layers with stride=2
        conv_output_size *= self.state_shape[2] // 4  # two conv layers with stride=2
        conv_output_size *= 16  # output channels of last conv layer

        # Feature extraction layers (shared)
        features = nn.Sequential(
            nn.Conv2d(self.state_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
        )

        # Advantage estimator
        advantage = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_space),
        )

        # Value estimator
        value = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        return features, advantage, value

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
        state_action_values = self.model(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.model(next_states).max(1)[0].detach()

        # Compute the expected Q values.
        expected_state_action_values = (next_state_values * self.gamma + rewards) * (
            1 - dones
        )

        # Update the weights.
        self._update_weights(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:  # epsilon is adjusted to often!!!git
            self.epsilon *= self.gamma

    def _update_weights(self, state_action_values, expected_state_action_values):
        # self.optimizers().zero_grad()
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        loss.backward()
        # self.optimizers().step()

    def forward(self, x):
        x = self.features(x)

        value = self.value(x)
        advantage = self.advantage(x)

        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.alpha)

    def training_step(self, batch, batch_idx):
        # This function is intentionally left blank, because we'll manually update the weights in replay()
        pass

    def load(self, name) -> None:
        self.load_state_dict(torch.load(name))

    def save(self, name) -> None:
        torch.save(self.state_dict(), name)
