import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Final

import numpy as np
from agents.dqn_torch import DQN
from gym.wrappers.monitoring import video_recorder
from loguru import logger
from pong_wrapper import PongWrapper
from utils.csv_logger import CsvLogger
from utils.file_utils import empty_directories

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# hyperparameters
MAX_EPISODES: Final[int] = 5000
FRAME_SKIP: Final[int] = 4
GAMMA: Final[float] = 0.99
LEARNING_RATE: Final[float] = 0.001
MEMORY_SIZE: Final[int] = 10_000
BATCH_SIZE: Final[int] = 64
EPSILON_DECAY: Final[float] = 0.999
EPSILON_MIN: Final[float] = 0.1
MODEL_SAVE_INTERVAL: Final[int] = 64
RECORD_INTERVAL: Final[int] = 512
STEP_PENALTY: Final[float] = 0.01
TARGET_NETWORK_UPDATE_INTERVAL: Final[int] = 1000
NUM_STACKED_FRAMES: Final[int] = 4
INPUT_DIM: Final[int] = 80
INPUT_SHAPE: Final[tuple] = (1, INPUT_DIM * NUM_STACKED_FRAMES, INPUT_DIM)

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")


csv_logger = CsvLogger(
    LOG_DIR / "training_metrics.log",
    [
        "episode",
        "reward",
        "steps",
        "epsilon",
        "time",
    ],
)


def log_episode(episode, reward, steps, epsilon, start_time):
    time_ = time() - start_time
    log_message = f"{episode},{reward:.2f},{steps:.2f},{epsilon:.4f},{time_:.2f}"

    logger.info(log_message)
    csv_logger.log(
        {
            "episode": episode,
            "reward": reward,
            "steps": steps,
            "epsilon": epsilon,
            "time": time_,
        }
    )


def loop():
    total_steps = 0
    epsilon = 1.0
    memory = deque(maxlen=MEMORY_SIZE)

    env = PongWrapper(
        "ALE/Pong-v5",
        skip=FRAME_SKIP,
        step_penalty=STEP_PENALTY,
        num_stacked_frames=NUM_STACKED_FRAMES,
    )

    # create the policy network
    dqn_policy = DQN(INPUT_SHAPE, num_actions=env.action_space.n)  # type: ignore

    # create the target network
    dqn_target = deepcopy(dqn_policy)

    # run main loop
    for episode in range(MAX_EPISODES):
        state = env.reset()

        episode_reward = 0
        episode_length = 0
        start_time = time()

        # set up the video recorder
        video = None
        if episode % RECORD_INTERVAL == 0:
            video = video_recorder.VideoRecorder(env, f"video/episode_{episode}.mp4")

        # run episode
        done = False
        while not done:
            total_steps += 1
            episode_length += 1
            if video:
                video.capture_frame()

            # act & observe
            action = dqn_policy.act(state, epsilon)
            next_state, reward, done = env.step(action)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # update policy network
            if len(memory) >= BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                minibatch = map(np.array, zip(*minibatch))
                dqn_policy.update(*minibatch, dqn_target)

            # periodically update the target network
            if total_steps % TARGET_NETWORK_UPDATE_INTERVAL == 0:
                dqn_target.copy_from(dqn_policy)

        # update epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # log episode
        log_episode(episode, episode_reward, episode_length, epsilon, start_time)

        # periodically save model
        if episode % MODEL_SAVE_INTERVAL == 0:
            dqn_policy.save_model(CHECKPOINTS_DIR / f"pong_model_{total_steps}.pth")

        # close the video recorder
        if video:
            video.close()


if __name__ == "__main__":
    empty_directories("checkpoints", "log", "video")
    loop()
