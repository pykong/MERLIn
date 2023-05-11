from typing import Final, NamedTuple, Self, Set

import cv2 as cv
import gym
import numpy as np
from gym.spaces import Discrete

__all__ = ["PongWrapper"]


class Step(NamedTuple):
    state: np.ndarray
    reward: float
    don: bool


def preprocess_state(state):
    """Shapes the observation space."""
    state = state[35:195]  # crop irrelevant parts of the image (top and bottom)
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)  # convert to grayscale
    state = cv.resize(state, (80, 80), interpolation=cv.INTER_AREA)  # downsample
    state = np.expand_dims(state, axis=0)  # add channel dimension at the beginning
    return state


class PongWrapper(gym.Wrapper):
    # https://gymnasium.farama.org/environments/atari/pong/#actions
    default_action: Final[int] = 0
    allowed_actions: Final[Set[int]] = {0, 1, 2, 3}  # TODO is '1' needed?

    def __init__(self: Self, env_name: str, skip: int = 1, step_penalty: float = 0):
        env = gym.make(env_name, render_mode="rgb_array")
        super().__init__(env)
        self.action_space = Discrete(len(self.allowed_actions))
        self.skip = skip
        self.step_penalty = step_penalty

    def step(self: Self, action: int) -> Step:
        action = self.default_action if action not in self.allowed_actions else action

        total_reward = 0
        next_state = None
        reward = 0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, _, _ = super().step(action)

            if reward == 0:
                # reward shaping
                total_reward -= self.step_penalty
            else:
                total_reward += reward

            if done:
                break

        return Step(preprocess_state(next_state), total_reward, done)

    def reset(self: Self):
        return preprocess_state(self.env.reset()[0])
