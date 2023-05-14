import random
from typing import Final

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from utils.replay_memory import Experience, ReplayMemory


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        gpu0 = torch.cuda.get_device_name(0)
        print(f"CUDA is available, running PyTorch on ({gpu0}).")
        return torch.device("cuda")
    else:
        print(f"Running PyTorch on CPU.")
        return torch.device("cpu")


class DQNSimpleAgent(L.LightningModule):
    name: Final[str] = "dqn_simple"

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
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.model: nn.Sequential = self._build_model()
        self.gpu = get_torch_device()
        self.model.to(self.gpu)

    def _build_model(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.state_shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.gpu)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.gpu)
            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                q_future = self.model(next_state).max(1)[0].item()
                target[0][action] = reward + q_future * self.epsilon_decay
            self._update_weights(state, target)

    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:  # epsilon is adjusted to often!!!git
            self.epsilon *= self.epsilon_decay

    def _update_weights(self, state, target):
        # self.optimizer.zero_grad()
        loss = F.mse_loss(self.model(state), target)
        loss.backward()
        # self.optimizer.step()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        # This function is intentionally left blank, because we'll manually update the weights in replay()
        pass

    def load(self, name) -> None:
        self.load_state_dict(torch.load(name))

    def save(self, name) -> None:
        torch.save(self.state_dict(), name)
