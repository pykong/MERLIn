import os
import random
import sys
from pathlib import Path
from typing import Final

import cv2 as cv
import numpy as np
import torch
from gym.wrappers.monitoring import video_recorder as vr

from .agents import DqnBaseAgent, agent_registry
from .config import Config
from .envs import BaseEnvWrapper, env_registry
from .memory import Transition
from .nets import BaseNet, net_registry
from .utils.file_utils import ensure_empty_dirs
from .utils.logging import EpisodeLog, EpisodeLogger, LogLevel
from .utils.silence_stdout import silence_stdout

# set random seeds for reproducibility
RANDOM_SEED: Final[int] = 0
np.random.seed(RANDOM_SEED)


def take_picture_of_state(state: np.ndarray, f_name: Path) -> None:
    state_transposed = np.transpose(state, (1, 2, 0))
    state_transposed *= 255  # increase brightness
    cv.imwrite(str(f_name), state_transposed)


def make_env(name: str, **kwargs) -> BaseEnvWrapper:
    """Create environment wrapper of provided name.

    Args:
        name (str): The identifier string of the environment wrapper.

    Returns:
        BaseEnvWrapper: A wrapper instance of the environment.
    """
    env_ = [e for e in env_registry if e.name == name][0]
    return env_(**kwargs)


def make_net(name: str) -> BaseNet:
    """Create neural net of provided name.

    Args:
        name (str): The identifier string of the neural network.

    Returns:
        BaseNet: The neural network instance.
    """
    net = [net for net in net_registry if net.name == name][0]
    return net()


def make_agent(agent_name: str, net_name: str, **kwargs) -> DqnBaseAgent:
    """Create agent of provided name and inject neural network.

    Args:
        agent_name (str): The identifier string of the agent.
        net_name (str): The identifier string of the neural network.

    Returns:
        DqnBaseAgent: The agent instance.
    """
    agent_ = [a for a in agent_registry if a.name == agent_name][0]
    kwargs["net"] = make_net(net_name)  # TODO: This is dirty
    return agent_(**kwargs)


def run_episode(
    agent,
    env: BaseEnvWrapper,
    episode_log: EpisodeLog,
    recorder: vr.VideoRecorder | None,
    img_dir: Path,
    save_img: bool = False,
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
        if save_img and random.choices([True, False], [1, 512], k=1)[0]:
            img_file = img_dir / f"{episode_log.episode}_{episode_log.steps}.png"
            take_picture_of_state(state, img_file)


def loop(config: Config, result_dir: Path):
    # define and prepare result dirs
    model_dir: Final[Path] = result_dir / "model"
    video_dir: Final[Path] = result_dir / "video"
    img_dir: Final[Path] = result_dir / "img"
    ensure_empty_dirs(model_dir, video_dir, img_dir)

    # calculate input shape
    input_shape: Final[tuple[int, int, int]] = (
        1,
        config.input_dim * config.num_stacked_frames,
        config.input_dim,
    )

    # configure torch
    torch.autograd.set_detect_anomaly(False)  # type: ignore
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.profiler.profile(enabled=False)

    # create environment
    env = make_env(
        config.env_name,
        state_dims=(config.input_dim, config.input_dim),
        skip=config.frame_skip,
        step_penalty=config.step_penalty,
        stack_size=config.num_stacked_frames,
    )

    # create the policy network
    agent: DqnBaseAgent = make_agent(
        config.agent_name,
        config.net_name,
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
    logger = EpisodeLogger(log_file=result_dir / f"{env.name}_{agent.name}.csv")

    # run main loop
    for episode in range(1, config.episodes + 1):
        # init episode logger
        episode_log = EpisodeLog(episode=episode, epsilon=agent.epsilon)
        episode_log.start_timer()

        # set up the video recorder
        recorder = None
        if episode % config.video_record_interval == 0:
            video_path = str(video_dir / f"{env.name}_{agent.name}_{episode}.mp4")
            logger.log(f"Recording video: {video_path}", LogLevel.VIDEO)
            recorder = vr.VideoRecorder(env, video_path)

        # run episode
        run_episode(agent, env, episode_log, recorder, img_dir, config.save_state_img)

        # log episode
        episode_log.stop_timer()
        logger.log(episode_log)

        # update epsilon
        if episode >= config.epsilon_decay_start:
            # TODO: Implement some form of logging
            # FIXME: The epsilon update is messed up, shared between loop and agent
            agent.update_epsilon(config.epsilon_step)

        # save model
        if episode > 0 and (
            (config.model_save_interval and episode % config.model_save_interval == 0)
            or episode == config.episodes  # always save at end of epoch
        ):
            model_name = f"{env.name}__{agent.name}__{config.net_name}__{episode}.pth"
            model_file = model_dir / model_name
            logger.log(f"Saving model: {model_file}", LogLevel.SAVE)
            agent.save(model_file)

        # close the video recorder
        if recorder:
            # shut the f*ck up, moviepy!
            with silence_stdout():
                recorder.close()
