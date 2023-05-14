from pathlib import Path
from time import time
from typing import Final

import numpy as np

# from agents.dqn_torch import DQN
from agents.dqn_torch_simply import DQNSimpleAgent
from gym.wrappers.monitoring import video_recorder as vr
from pong_wrapper import PongWrapper
from utils.file_utils import empty_directories
from utils.logging import EpisodeLog, log_to_csv
from utils.replay_memory import Experience

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")
VIDEO_DIR: Final[Path] = Path("video")

# hyperparameters
MAX_EPISODES: Final[int] = 5000
FRAME_SKIP: Final[int] = 4
LEARNING_RATE: Final[float] = 0.001
MEMORY_SIZE: Final[int] = 10_000
BATCH_SIZE: Final[int] = 64
EPSILON_DECAY: Final[float] = 0.999  # discount factor gamma
EPSILON_MIN: Final[float] = 0.1
MODEL_SAVE_INTERVAL: Final[int] = 64
RECORD_INTERVAL: Final[int] = 8
STEP_PENALTY: Final[float] = 0.01
TARGET_NETWORK_UPDATE_INTERVAL: Final[int] = 1000
NUM_STACKED_FRAMES: Final[int] = 4
INPUT_DIM: Final[int] = 80
INPUT_SHAPE: Final[tuple] = (INPUT_DIM * NUM_STACKED_FRAMES, INPUT_DIM, 1)


def loop():
    # create environment
    env = PongWrapper(
        "ALE/Pong-v5",
        skip=FRAME_SKIP,
        step_penalty=STEP_PENALTY,
        stack_size=NUM_STACKED_FRAMES,
    )

    # create the policy network
    agent = DQNSimpleAgent(
        state_shape=INPUT_SHAPE,
        action_space=env.action_space.n,  # type: ignore
        epsilon_decay=EPSILON_DECAY,
        alpha=LEARNING_RATE,
    )

    # run main loop
    for episode in range(MAX_EPISODES):
        state = env.reset()

        episode_log = EpisodeLog(episode=episode, epsilon=agent.epsilon)
        start_time = time()

        # set up the video recorder
        video = None
        if episode % RECORD_INTERVAL == 0:
            video_path = str(VIDEO_DIR / f"{env.name}_{agent.name}_{episode}.mp4")
            video = vr.VideoRecorder(env, video_path)

        # run episode
        done = False
        while not done:
            # prepare step
            episode_log.steps += 1
            if video:
                video.capture_frame()

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

        # update epsilon
        agent.update_epsilon()

        # log episode
        episode_log.time = time() - start_time
        print(episode_log)
        log_to_csv(episode_log, LOG_DIR / f"{env.name}_{agent.name}.csv")

        # periodically save model
        if episode % MODEL_SAVE_INTERVAL == 0:
            agent.save(CHECKPOINTS_DIR / f"{env.name}_{agent.name}_{episode}.pth")

        # close the video recorder
        if video:
            video.close()


if __name__ == "__main__":
    empty_directories(CHECKPOINTS_DIR, LOG_DIR, VIDEO_DIR)
    loop()
