import random
from collections import deque
from pathlib import Path
from time import time
from typing import Final

import numpy as np
from dqn_torch import DQN
from loguru import logger
from pong_wrapper import PongWrapper
from utils.csv_logger import CsvLogger

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# hyperparameters
MAX_EPISODES: Final[int] = 20_000
GAMMA: Final[float] = 0.99
LEARNING_RATE: Final[float] = 0.001
MEMORY_SIZE: Final[int] = 100_000
BATCH_SIZE: Final[int] = 64
EPSILON_DECAY: Final[float] = 0.999
EPSILON_MIN: Final[float] = 0.1
MODEL_SAVE_INTERVAL: Final[int] = 1  # episode
LOG_INTERVAL: Final[int] = 1

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")


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
        LOG_DIR / "training_metrics.log",
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
        state = env.reset()

        done = False
        episode_reward = 0
        episode_length = 0
        start_time = time()

        # run episode
        while not done:
            action = dqn.act(state, epsilon)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            episode_reward += reward
            episode_length += 1

            if len(memory) >= BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                minibatch = map(np.array, zip(*minibatch))
                dqn.update(*minibatch)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)  # update epsilon

        episode_time.append(time() - start_time)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 0:
            win_count += 1

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
            dqn.save_model(CHECKPOINTS_DIR / f"pong_model_{total_steps}.pth")


if __name__ == "__main__":
    loop()
