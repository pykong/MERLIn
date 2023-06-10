import os
import random
import re
import sys
from pathlib import Path
from typing import Final

import cv2 as cv
import numpy as np
from agents.base_agent import BaseAgent
from agents.dqn_double import DoubleDQNAgent
from agents.dqn_duelling import DuellingDQNAgent
from agents.dqn_vanilla import VanillaDQNAgent
from config import Config
from gym.wrappers.monitoring import video_recorder as vr
from pong_wrapper import PongWrapper
from utils.file_utils import ensure_empty_dirs
from utils.logging import EpisodeLog, EpisodeLogger, LogLevel, logger
from utils.replay_memory import Transition

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)

# checkpoints dir
RESULTS_DIR: Final[Path] = Path("results")
CHECKPOINTS_DIR: Final[Path] = RESULTS_DIR / "checkpoints"
LOG_DIR: Final[Path] = RESULTS_DIR / "log"
VIDEO_DIR: Final[Path] = RESULTS_DIR / "video"
IMG_DIR: Final[Path] = RESULTS_DIR / "img"


def take_picture_of_state(state: np.ndarray, f_name: Path) -> None:
    state_transposed = np.transpose(state, (1, 2, 0))
    state_transposed *= 255  # enhance pixel brightness
    cv.imwrite(str(f_name), state_transposed)


def make_agent(name: str, load_agent: bool, **kwargs) -> BaseAgent:
    """Factory method to create agent."""
    registry = [VanillaDQNAgent, DoubleDQNAgent, DuellingDQNAgent]
    agent_ = [a for a in registry if a.name == name][0]
    agent = agent_(**kwargs)
    if load_agent:
        models = CHECKPOINTS_DIR.glob("*.pth")
        env = "pong"  # TODO: Dynamize env name
        pattern = re.compile(f"{env}_{name}_\\d+\\.pth")
        models = [m for m in models if pattern.match(m.name)]
        if not models:
            logger.log(str(LogLevel.GREEN), f"No checkpoint found for: {name}")
        else:
            sorted(models)
            latest_model = models[0]
            agent.load(latest_model)
            logger.log(str(LogLevel.GREEN), f"Loading checkpoint: {latest_model.name}")
    return agent


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
        episode_log.loss += agent.replay()

        # ???
        state = next_state
        episode_log.reward += reward

        # take picture of state randomly
        if config.save_state_img and random.choices([True, False], [1, 512], k=1)[0]:
            img_file = IMG_DIR / f"{episode_log.episode}_{episode_log.steps}.png"
            take_picture_of_state(state, img_file)


def loop(config: Config):
    # suppress moviepy output: ultimata ratio :-|
    if not config.verbose:
        sys.stdout = open(os.devnull, "w")

    # calculate input shape
    input_shape: Final[tuple[int, int, int]] = (
        1,
        config.input_dim * config.num_stacked_frames,
        config.input_dim,
    )

    # create environment
    env = PongWrapper(
        "ALE/Pong-v5",
        state_dims=(config.input_dim, config.input_dim),
        skip=config.frame_skip,
        step_penalty=config.step_penalty,
        stack_size=config.num_stacked_frames,
    )

    # create the policy network
    agent: BaseAgent = make_agent(
        config.agent_name,
        config.load_agent,
        state_shape=input_shape,
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
    for episode in range(1, config.max_episodes + 1):
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

        # update epsilon
        if episode >= config.start_epsilon_decay:
            # TODO: Implement some form of logging
            agent.update_epsilon()

        # save model
        if episode > 0 and (
            episode % config.model_save_interval == 0 or episode == config.max_episodes
        ):
            model_file = CHECKPOINTS_DIR / f"{env.name}_{agent.name}_{episode}.pth"
            logger.log(f"Saving model: {model_file}", LogLevel.SAVE)
            agent.save(model_file)

        # close the video recorder
        if recorder:
            recorder.close()


if __name__ == "__main__":
    ensure_empty_dirs(CHECKPOINTS_DIR, LOG_DIR, VIDEO_DIR, IMG_DIR)
    config = Config()
    loop(config)
