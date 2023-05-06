import random
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(0)


# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
MODEL_SAVE_INTERVAL = 1000
LOG_INTERVAL = 10


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
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
            return torch.argmax(
                self(torch.Tensor(np.expand_dims(state, axis=0)))
            ).item()


def preprocess_state(state):
    state = state[35:195]  # Crop the irrelevant parts of the image (top and bottom)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(
        state, (80, 80), interpolation=cv2.INTER_AREA
    )  # Downsample to 80x80
    state = state[
        :, :, np.newaxis
    ]  # Add a channel dimension for the neural network input
    return state


def main():
    env = gym.make("PongDeterministic-v4")
    input_shape = (80, 80, 1)
    num_actions = env.action_space.n

    dqn = DQN(input_shape, num_actions)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    win_count = 0

    with open("training_metrics.log", "w") as log_file:
        log_file.write("episode,avg_reward,win_rate,avg_episode_length,epsilon\n")

        episode = 0
        while True:
            episode += 1
            state = preprocess_state(env.reset()[0])
            done = False

            episode_reward = 0
            episode_length = 0

            while not done:
                action = dqn.act(state, epsilon)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_state(next_state)
                memory.append((state, action, reward, next_state, done))
                state = next_state

                episode_reward += reward
                episode_length += 1

                if len(memory) >= BATCH_SIZE:
                    minibatch = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = map(
                        np.array, zip(*minibatch)
                    )
                    dqn.update(states, actions, rewards, next_states, dones)

                total_steps += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if episode_reward > 0:
                    win_count += 1
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

                if episode % LOG_INTERVAL == 0:
                    avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
                    avg_episode_length = np.mean(episode_lengths[-LOG_INTERVAL:])
                    win_rate = (win_count / LOG_INTERVAL) * 100

                    log_message = f"{episode},{avg_reward:.2f},{win_rate:.2f},{avg_episode_length:.2f},{epsilon:.4f}"
                    print(log_message)
                    log_file.write(log_message + "\n")
                    win_count = 0

                if total_steps % MODEL_SAVE_INTERVAL == 0:
                    torch.save(dqn.state_dict(), f"pong_model_{total_steps}.pth")


if __name__ == "__main__":
    main()
