import os
import random
import sys
from pathlib import Path
from typing import Final

import cv2 as cv
import numpy as np
# from agents.dqn_cnn_double import DDQNCNNAgent

from agents.dqn_cnn_duelling import DuellingQNCNNAgent
from gym.wrappers.monitoring import video_recorder as vr
from pong_wrapper import PongWrapper
from utils.file_utils import empty_directories
from utils.logging import EpisodeLog, EpisodeLogger, LogLevel
from utils.replay_memory import Experience

# from agents.dqn_linear_double import DDQNLinearAgent


# from agents.dqn_torch import DQN
# from agents.dqn_cnn import DQNCNNAgent


# suppress moviepy output: ultimata ratio :-|
sys.stdout = open(os.devnull, "w")

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")
VIDEO_DIR: Final[Path] = Path("video")
IMG_DIR: Final[Path] = Path("img")

# hyperparameters
MAX_EPISODES: Final[int] = 20_000
FRAME_SKIP: Final[int] = 4
LEARNING_RATE: Final[float] = 5e-3
MEMORY_SIZE: Final[int] = 64_000
BATCH_SIZE: Final[int] = 64
EPSILON_DECAY: Final[float] = 1 - 1e-4  # discount factor gamma
EPSILON_MIN: Final[float] = 0.1
MODEL_SAVE_INTERVAL: Final[int] = 1024
RECORD_INTERVAL: Final[int] = 512
STEP_PENALTY: Final[float] = 0.01
TARGET_NETWORK_UPDATE_INTERVAL: Final[int] = 8
NUM_STACKED_FRAMES: Final[int] = 2
INPUT_DIM: Final[int] = 80
INPUT_SHAPE: Final[tuple[int, int, int]] = (
    1,
    INPUT_DIM * NUM_STACKED_FRAMES,
    INPUT_DIM,
)
SAVE_STATE_IMG: Final[bool] = False


def take_picture_of_state(state: np.ndarray, f_name: Path) -> None:
    state_transposed = np.transpose(state, (1, 2, 0))
    state_transposed *= 255  # enhance pixel brightness
    cv.imwrite(str(f_name), state_transposed)


def loop():
    # create environment
    env = PongWrapper(
        "ALE/Pong-v5",
        state_dims=(INPUT_DIM, INPUT_DIM),
        skip=FRAME_SKIP,
        step_penalty=STEP_PENALTY,
        stack_size=NUM_STACKED_FRAMES,
    )

    # create the policy network
    agent = DuellingQNCNNAgent(
        state_shape=INPUT_SHAPE,
        action_space=env.action_space.n,  # type: ignore
        gamma=EPSILON_DECAY,
        alpha=LEARNING_RATE,
    )

    # init logger
    logger = EpisodeLogger(log_file=LOG_DIR / f"{env.name}_{agent.name}.csv")

    # run main loop
    for episode in range(MAX_EPISODES):
        # reset environment
        state = env.reset()

        # init episode logger
        episode_log = EpisodeLog(episode=episode, epsilon=agent.epsilon)
        episode_log.start_timer()

        # set up the video recorder
        recorder = None
        if episode % RECORD_INTERVAL == 0:
            video_path = str(VIDEO_DIR / f"{env.name}_{agent.name}_{episode}.mp4")
            logger.log(f"Recording video: {video_path}", LogLevel.VIDEO)
            recorder = vr.VideoRecorder(env, video_path)

        # run episode
        done = False
        while not done:
            # prepare step
            episode_log.steps += 1
            if recorder:
                recorder.capture_frame()

            # act & observe
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # save experience
            experience = Experience(state, action, reward, next_state, done)
            agent.remember(experience)

            # update policy network
            agent.replay()

            # ???
            state = next_state
            episode_log.reward += reward

            # take picture of state randomly
            if SAVE_STATE_IMG and random.choices([True, False], [1, 512], k=1)[0]:
                img_file = IMG_DIR / f"{episode}_{episode_log.steps}.png"
                take_picture_of_state(state, img_file)

        # update epsilon
        agent.update_epsilon()

        # log episode
        episode_log.stop_timer()
        logger.log(episode_log)

        # update target network
        # if episode % TARGET_NETWORK_UPDATE_INTERVAL == 0:
        #     logger.log(f"Updating target network", LogLevel.GREEN)
        #     agent.update_target()

        # periodically save model
        if episode % MODEL_SAVE_INTERVAL == 0:
            model_file = CHECKPOINTS_DIR / f"{env.name}_{agent.name}_{episode}.pth"
            logger.log(f"Saving model: {model_file}", LogLevel.SAVE)
            agent.save(model_file)

        # close the video recorder
        if recorder:
            recorder.close()


if __name__ == "__main__":
    empty_directories(CHECKPOINTS_DIR, LOG_DIR, VIDEO_DIR, IMG_DIR)
    loop()
