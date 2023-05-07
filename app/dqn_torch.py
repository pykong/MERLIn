from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(0)


# Hyperparameters
MAX_EPISODES = 20_000
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
MODEL_SAVE_INTERVAL = 50  # episode
LOG_INTERVAL = 10


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)

        # Calculate the output dimensions of the last convolutional layer
        def conv_output_dim(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv_output_dim(
            conv_output_dim(conv_output_dim(input_shape[1], 8, 4), 4, 2), 3, 1
        )
        convh = conv_output_dim(
            conv_output_dim(conv_output_dim(input_shape[2], 8, 4), 4, 2), 3, 1
        )

        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def update(self, states, actions, rewards, next_states, dones):
        target_values = self(torch.Tensor(states))
        next_target_values = self(torch.Tensor(next_states)).detach()
        target_values[np.arange(states.shape[0]), actions] = torch.Tensor(
            rewards
        ) + GAMMA * torch.max(next_target_values, dim=1).values * torch.Tensor(
            1 - dones
        )

        self.optimizer.zero_grad()
        loss = self.loss_fn(self(torch.Tensor(states)), target_values)
        loss.backward()
        self.optimizer.step()

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.fc2.out_features)
        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0)
            )  # Convert state to a float tensor and add a batch dimension
            return torch.argmax(self(state_tensor)).item()

    def save_model(self, file_name: Path) -> None:
        file_name.parent.mkdir(parents=True, exist_ok=True)  # ensure folders
        torch.save(self.state_dict(), file_name)
