import random
from collections import deque, namedtuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        gpu0 = torch.cuda.get_device_name(0)
        logger.success(f"CUDA is available, running PyTorch on ({gpu0}).")
        return torch.device("cuda")
    else:
        logger.warn(f"Running PyTorch on CPU.")
        return torch.device("cpu")


class DQNSimpleAgent(L.LightningModule):
    def __init__(
        self,
        state_shape,
        action_space,
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.Experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.model = self._build_model()
        self.gpu = get_torch_device()
        self.model.to(self.gpu)

    def _build_model(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.state_shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.gpu)
        act_values = self.model(state)
        return int(torch.argmax(act_values[0]).item())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.gpu)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.gpu)
            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = self.model(next_state).max(1)[0].item()
                target[0][action] = reward + Q_future * self.gamma
            self._update_weights(state, target)
        if self.epsilon > self.epsilon_min:
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

    def load(self, name):
        self.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.state_dict(), name)
