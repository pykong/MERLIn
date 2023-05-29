from collections import deque
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
    state = np.expand_dims(state, axis=0)  # prepend channel dimension
    return state


class PongWrapper(gym.Wrapper):
    # https://gymnasium.farama.org/environments/atari/pong/#actions
    name: Final[str] = "pong"
    default_action: Final[int] = 0
    valid_actions: Final[Set[int]] = {0, 1, 2, 3}

    def __init__(
        self: Self,
        env_name: str,
        skip: int = 1,
        step_penalty: float = 0,
        stack_size: int = 1,
    ):
        env = gym.make(env_name, render_mode="rgb_array")
        env.metadata["render_fps"] = 25
        super().__init__(env)
        self.action_space = Discrete(len(self.valid_actions))
        self.skip = skip
        self.step_penalty = step_penalty
        self.stack_size = stack_size
        self.state_buffer = deque([], maxlen=self.stack_size)

    def step(self: Self, action: int) -> Step:
        action = self.default_action if action not in self.valid_actions else action

        total_reward = 0
        next_state = None
        reward = 0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, _, _ = super().step(action)
            total_reward += reward
            if done:
                break

        if total_reward == 0:
            total_reward = -self.step_penalty

        self.state_buffer.append(preprocess_state(next_state))
        stacked_state = np.concatenate(self.state_buffer, axis=1)
        return Step(stacked_state, total_reward, done)

    def reset(self: Self) -> np.ndarray:
        state = preprocess_state(self.env.reset()[0])
        self.state_buffer = deque([state] * self.stack_size, maxlen=self.stack_size)
        return np.concatenate(self.state_buffer, axis=1)
