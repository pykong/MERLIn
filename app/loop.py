import random
from pathlib import Path
from typing import Final

import cv2 as cv
import numpy as np
import torch
from gym.wrappers.monitoring import video_recorder as vr

from app.agents import DqnBaseAgent, make_agent
from app.config import Config
from app.envs import BaseEnvWrapper, make_env
from app.memory import Transition
from app.nets import BaseNet, make_net
from app.utils.file_utils import ensure_empty_dirs
from app.utils.logging import EpisodeLog, EpisodeLogger, LogLevel
from app.utils.silence_stdout import silence_stdout


def take_picture_of_state(state: np.ndarray, f_name: Path) -> None:
    """Save brightened picture of current state to file.

    Args:
        state (np.ndarray): The state to taken picture of.
        f_name (Path): The file path to save to.
    """
    state_transposed = np.transpose(state, (1, 2, 0))
    state_transposed *= 255  # type:ignore | increase brightness
    cv.imwrite(str(f_name), state_transposed)


def run_episode(
    agent: DqnBaseAgent,
    env: BaseEnvWrapper,
    episode_log: EpisodeLog,
    recorder: vr.VideoRecorder | None,
    img_dir: Path,
    save_img: bool = False,
) -> None:
    """Run single episode.

    Args:
        agent (DqnBaseAgent): The agent instance.
        env (BaseEnvWrapper): The environment instance.
        episode_log (EpisodeLog): The episode logger instance.
        recorder (vr.VideoRecorder | None): The video recorder instance.
        img_dir (Path): Path to save images to.
        save_img (bool, optional): Whether to save image states. Defaults to False.
    """

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


def loop(config: Config, result_dir: Path) -> None:
    """Run all episodes.

    Args:
        config (Config): The configuration object, holding the experiment parameters.
        result_dir (Path): The dir to save experiment results to.
    """
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

    # set seed for reproducibility
    np.random.seed(config.run)

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
        net=make_net(config.net_name),
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
    logger = EpisodeLogger(log_file=result_dir / "train_log.csv")

    # run main loop
    for episode in range(1, config.episodes + 1):
        # init episode logger
        episode_log = EpisodeLog(
            episode=episode,
            epsilon=agent.epsilon,
            experiment=config.experiment,
            variant=config.variant,
            run=config.run,
        )
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
            model_file = model_dir / f"{episode}.pth"
            logger.log(f"Saving model: {model_file}", LogLevel.SAVE)
            agent.save(model_file)

        # close the video recorder
        if recorder:
            # shut the f*ck up, moviepy!
            with silence_stdout():
                recorder.close()
