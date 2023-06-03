import os
import random
import sys
from pathlib import Path
from typing import Final

import cv2 as cv
import numpy as np
from agents.base_agent import BaseAgent
from agents.dqn_cnn_double import DDQNCNNAgent
from config import Config
from gym.wrappers.monitoring import video_recorder as vr
from pong_wrapper import PongWrapper
from utils.file_utils import empty_directories
from utils.logging import EpisodeLog, EpisodeLogger, LogLevel
from utils.replay_memory import Transition

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# checkpoints dir
CHECKPOINTS_DIR: Final[Path] = Path("checkpoints")
LOG_DIR: Final[Path] = Path("log")
VIDEO_DIR: Final[Path] = Path("video")
IMG_DIR: Final[Path] = Path("img")

# hyperparameters
NUM_STACKED_FRAMES: Final[int] = 1
INPUT_DIM: Final[int] = 80
INPUT_SHAPE: Final[tuple[int, int, int]] = (
    1,
    INPUT_DIM * NUM_STACKED_FRAMES,
    INPUT_DIM,
)


def take_picture_of_state(state: np.ndarray, f_name: Path) -> None:
    state_transposed = np.transpose(state, (1, 2, 0))
    state_transposed *= 255  # enhance pixel brightness
    cv.imwrite(str(f_name), state_transposed)


def make_agent(name: str, load_agent: bool, **kwargs) -> BaseAgent:
    """Factory method to create agent."""
    registry = [DDQNCNNAgent]
    agent = [a for a in registry if a.name == name][0]
    if load_agent:
        models = CHECKPOINTS_DIR.glob("*.pth")
        models = [m for m in models if name in m.name]
        print(f"found models: {models}")
    return agent(**kwargs)


def run_episode(
    agent, env: PongWrapper, episode_log: EpisodeLog, recorder: vr.VideoRecorder | None
) -> None:
    # reset environment
    state = env.reset()

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
        transition = Transition(state, action, reward, next_state, done)
        agent.remember(transition)

        # update policy network
        agent.replay()

        # ???
        state = next_state
        episode_log.reward += reward

        # take picture of state randomly
        if config.save_state_img and random.choices([True, False], [1, 512], k=1)[0]:
            img_file = IMG_DIR / f"{episode_log.episode}_{episode_log.steps}.png"
            take_picture_of_state(state, img_file)

    # update epsilon
    agent.update_epsilon()


def loop(config: Config):
    # suppress moviepy output: ultimata ratio :-|
    if not config.verbose:
        sys.stdout = open(os.devnull, "w")

    # create environment
    env = PongWrapper(
        "ALE/Pong-v5",
        state_dims=(INPUT_DIM, INPUT_DIM),
        skip=config.frame_skip,
        step_penalty=config.step_penalty,
        stack_size=NUM_STACKED_FRAMES,
    )

    # create the policy network
    agent: BaseAgent = make_agent(
        config.agent_name,
        config.load_agent,
        state_shape=INPUT_SHAPE,
        action_space=env.action_space.n,  # type: ignore
        gamma=config.gamma,
        alpha=config.alpha,
        epsilon_min=config.epsilon_min,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_net_update_interval=config.target_net_update_interval,
    )

    # init logger
    logger = EpisodeLogger(log_file=LOG_DIR / f"{env.name}_{agent.name}.csv")

    # run main loop
    for episode in range(config.max_episodes):
        # init episode logger
        episode_log = EpisodeLog(episode=episode, epsilon=agent.epsilon)
        episode_log.start_timer()

        # set up the video recorder
        recorder = None
        if episode % config.video_record_interval == 0:
            video_path = str(VIDEO_DIR / f"{env.name}_{agent.name}_{episode}.mp4")
            logger.log(f"Recording video: {video_path}", LogLevel.VIDEO)
            recorder = vr.VideoRecorder(env, video_path)

        # run episode
        run_episode(agent, env, episode_log, recorder)

        # log episode
        episode_log.stop_timer()
        logger.log(episode_log)

        # periodically save model
        if episode % config.model_save_interval == 0:
            model_file = CHECKPOINTS_DIR / f"{env.name}_{agent.name}_{episode}.pth"
            logger.log(f"Saving model: {model_file}", LogLevel.SAVE)
            agent.save(model_file)

        # close the video recorder
        if recorder:
            recorder.close()


if __name__ == "__main__":
    empty_directories(LOG_DIR, VIDEO_DIR, IMG_DIR)  # CHECKPOINTS_DIR
    config = Config()
    loop(config)
