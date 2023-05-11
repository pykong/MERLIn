import random
from collections import deque
from pathlib import Path
from time import time
from typing import Final

import numpy as np
from dqn_torch import DQN
from gym.wrappers.monitoring import video_recorder
from loguru import logger
from pong_wrapper import PongWrapper
from utils.csv_logger import CsvLogger
from utils.torch_device import get_torch_device

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# hyperparameters
MAX_EPISODES: Final[int] = 5000
FRAME_SKIP: Final[int] = 4
GAMMA: Final[float] = 0.99
LEARNING_RATE: Final[float] = 0.001
MEMORY_SIZE: Final[int] = 100_000
BATCH_SIZE: Final[int] = 64
EPSILON_DECAY: Final[float] = 0.995
EPSILON_MIN: Final[float] = 0.1
MODEL_SAVE_INTERVAL: Final[int] = 64
RECORD_INTERVAL: Final[int] = 512
STEP_PENALTY: Final[float] = 0.01
TARGET_NETWORK_UPDATE_INTERVAL: Final[int] = 1000

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")


def loop():
    env = PongWrapper("ALE/Pong-v5", skip=FRAME_SKIP, step_penalty=STEP_PENALTY)
    input_shape = (1, 80, 80)
    num_actions = env.action_space.n  # type: ignore

    dqn = DQN(input_shape, num_actions)
    dqn.to(get_torch_device())

    # Create a target network
    dqn_target = DQN(input_shape, num_actions)
    dqn_target.to(get_torch_device())
    dqn_target.copy_from(dqn)

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    total_steps = 0

    csv_logger = CsvLogger(
        LOG_DIR / "training_metrics.log",
        [
            "episode",
            "episode_reward",
            "episode_length",
            "epsilon",
            "cpu_time",
        ],
    )

    for episode in range(MAX_EPISODES):
        state = env.reset()

        episode_reward = 0
        episode_length = 0
        start_time = time()

        video = None
        if episode % RECORD_INTERVAL == 0:
            # Set up the video recorder
            video = video_recorder.VideoRecorder(env, f"video/episode_{episode}.mp4")

        # run episode
        done = False
        while not done:
            total_steps += 1
            if video:
                video.capture_frame()
            action = dqn.act(state, epsilon)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            episode_reward += reward
            episode_length += 1

            if len(memory) >= BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                minibatch = map(np.array, zip(*minibatch))
                dqn.update(*minibatch, dqn_target)

            # Periodically update the target network
            if total_steps % TARGET_NETWORK_UPDATE_INTERVAL == 0:
                dqn_target.copy_from(dqn)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)  # update epsilon

        # log episode
        cpu_time = time() - start_time

        log_message = f"{episode},{episode_reward:.2f},{episode_length:.2f},{epsilon:.4f},{cpu_time:.2f}"

        logger.info(log_message)
        csv_logger.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "epsilon": epsilon,
                "cpu_time": cpu_time,
            }
        )

        if episode % MODEL_SAVE_INTERVAL == 0:
            dqn.save_model(CHECKPOINTS_DIR / f"pong_model_{total_steps}.pth")

        # Close the video recorder
        if video:
            video.close()


if __name__ == "__main__":
    loop()
