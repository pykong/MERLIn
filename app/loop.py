import random
from collections import deque
from csv import DictWriter
from pathlib import Path
from time import time
from typing import Self

import cv2 as cv
import numpy as np
from dqn_torch import DQN
from loguru import logger
from pong_wrapper import PongWrapper

# Set random seeds for reproducibility
np.random.seed(0)


# Hyperparameters
MAX_EPISODES = 20_000
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
MODEL_SAVE_INTERVAL = 1  # episode
LOG_INTERVAL = 1


def preprocess_state(state):
    """Shapes the observation space."""
    state = state[35:195]  # crop irrelevant parts of the image (top and bottom)
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)  # convert to grayscale
    state = cv.resize(state, (80, 80), interpolation=cv.INTER_AREA)  # downsample
    state = np.expand_dims(state, axis=0)  # add channel dimension at the beginning
    return state


class CsvLogger:
    def __init__(self: Self, log_file: Path, columns: list[str]) -> None:
        self.log_file = log_file
        self.columns = columns
        with open(self.log_file, "w", newline="") as csvfile:
            self.writer = DictWriter(csvfile, fieldnames=columns)
            self.writer.writeheader()

    def log(self: Self, items: dict[str, str | int | float]) -> None:
        with open(self.log_file, "a", newline="") as csvfile:
            self.writer = DictWriter(csvfile, fieldnames=self.columns)
            self.writer.writerow(items)


def loop():
    env = PongWrapper("PongDeterministic-v4")
    input_shape = (1, 80, 80)
    num_actions = env.action_space.n

    dqn = DQN(input_shape, num_actions)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_time = []
    win_count = 0

    csv_logger = CsvLogger(
        Path("training_metrics.log"),
        [
            "episode",
            "avg_reward",
            "win_rate",
            "avg_episode_length",
            "epsilon",
            "avg_time",
        ],
    )

    for episode in range(MAX_EPISODES):
        print(f"episode: {episode}")
        state = preprocess_state(env.reset()[0])
        done = False

        episode_reward = 0
        episode_length = 0

        # run episode
        start_time = time()
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

        episode_time.append(time() - start_time)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 0:
            win_count += 1
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % LOG_INTERVAL == 0:
            avg_time = np.mean(episode_time[-LOG_INTERVAL:])
            avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
            avg_episode_length = np.mean(episode_lengths[-LOG_INTERVAL:])
            win_rate = (win_count / LOG_INTERVAL) * 100

            log_message = f"{episode},{avg_reward:.2f},{win_rate:.2f},{avg_episode_length:.2f},{epsilon:.4f},{avg_time:.2f}"

            logger.info(log_message)
            csv_logger.log(
                {
                    "episode": episode,
                    "avg_reward": avg_reward,
                    "win_rate": win_rate,
                    "avg_episode_length": avg_episode_length,
                    "epsilon": epsilon,
                    "avg_time": avg_time,
                }
            )
            win_count = 0

        if episode % MODEL_SAVE_INTERVAL == 0:
            dqn.save_model(f"pong_model_{total_steps}.pth")


if __name__ == "__main__":
    loop()
