from abc import ABC, abstractmethod
from collections import deque
from typing import Self

import cv2 as cv
import gym
import numpy as np
from gym.spaces import Discrete

from .step import Step


class BaseEnvWrapper(gym.Wrapper, ABC):
    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        """String to represent wrapper to the outside."""
        raise NotImplementedError()

    @classmethod
    @property
    @abstractmethod
    def env_name(cls) -> str:
        """String to identify Atari environment."""
        raise NotImplementedError()

    @classmethod
    @property
    @abstractmethod
    def valid_actions(cls) -> set[int]:
        """Set of valid actions to chose."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _crop_state(cls, state: np.ndarray) -> np.ndarray:
        """Crop state to informative region."""
        # TODO: Make getter returning slice
        raise NotImplementedError()

    def __init__(
        self: Self,
        state_dims: tuple[int, int],
        skip: int = 1,
        step_penalty: float = 0,
        stack_size: int = 1,
    ):
        env = gym.make(self.env_name, render_mode="rgb_array")
        env.metadata["render_fps"] = 25
        super().__init__(env)
        self.state_dims = state_dims
        self.action_space = Discrete(len(self.valid_actions))
        self.skip = skip
        self.step_penalty = step_penalty
        self.stack_size = stack_size
        self.state_buffer = deque([], maxlen=self.stack_size)

    def step(self: Self, action_idx: int) -> Step:
        action = list(self.valid_actions)[action_idx]
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

        self.state_buffer.append(self.__preprocess_state(next_state, self.state_dims))
        stacked_state = self.__stack_frames(self.state_buffer)
        return Step(stacked_state, total_reward, done)

    def reset(self: Self) -> np.ndarray:
        state = self.__preprocess_state(self.env.reset()[0], self.state_dims)
        self.state_buffer = deque([state] * self.stack_size, maxlen=self.stack_size)
        return self.__stack_frames(self.state_buffer)

    @staticmethod
    def __stack_frames(state_buffer: deque) -> np.ndarray:
        return np.concatenate(state_buffer, axis=1)

    @classmethod
    def __preprocess_state(cls, state, state_dims: tuple[int, int]) -> np.ndarray:
        """Shapes the observation space."""
        state = cls._crop_state(state)
        state = cv.resize(state, state_dims, interpolation=cv.INTER_AREA)  # downsample
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)  # remove channrl dim
        # TODO: put threshold value into constant
        _, state = cv.threshold(state, 64, 255, cv.THRESH_BINARY)  # make binary
        state = cv.normalize(
            state,
            None,
            alpha=0,
            beta=1,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_32F,
        )
        state = np.expand_dims(state, axis=0)  # prepend channel dimension
        return state
