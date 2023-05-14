from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from loguru import logger

__all__ = ["DQN"]


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        gpu0 = torch.cuda.get_device_name(0)
        # logger.info(f"CUDA is available. Running PyTorch on GPU ({gpu0}).")
        return torch.device("cuda")
    else:
        # logger.info(f"Running PyTorch on CPU.")
        return torch.device("cpu")


class DQN(nn.Module):
    def __init__(
        self: Self,
        input_shape,
        *,
        num_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.001,
    ) -> None:
        """The deep Q network implementation.

        Args:
            self (Self): The instance.
            input_shape (tuple): The input shape.
            num_actions (int): The number of actions.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            alpha (float, optional): The learning rate alpha. Defaults to 0.001.

        Returns:
            _type_: _description_
        """
        super(DQN, self).__init__()
        self.gamma = gamma
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

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.to(get_torch_device())

    def forward(self: Self, x: torch.Tensor) -> torch.nn.Linear:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def update(
        self: Self, states, actions, rewards, next_states, dones, target_network
    ) -> None:
        device = next(self.parameters()).device  # Get the device of the model

        # Move tensors to the device and ensure they all have the same data type (torch.float32)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(
            device
        )  # Use long dtype for indices
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        target_values = self(states)
        next_target_values = target_network(next_states).detach()
        target_values[
            np.arange(states.shape[0]), actions
        ] = rewards + self.gamma * torch.max(next_target_values, dim=1).values * (
            1.0 - dones
        )

        self.optimizer.zero_grad()
        loss = self.loss_fn(self(states), target_values)
        loss.backward()
        self.optimizer.step()

    def act(self: Self, state, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # random exploration
            return np.random.randint(self.fc2.out_features)
        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0)
            )  # Convert state to a float tensor and add a batch dimension
            device = next(self.parameters()).device  # Get the device of the modelg
            state_tensor = state_tensor.clone().detach().to(device)
            return int(torch.argmax(self(state_tensor)).item())

    def save_model(self: Self, file_name: Path) -> None:
        file_name.parent.mkdir(parents=True, exist_ok=True)  # ensure folders
        torch.save(self.state_dict(), file_name)

    def copy_from(self: Self, other: "DQN") -> None:
        self.load_state_dict(other.state_dict())
