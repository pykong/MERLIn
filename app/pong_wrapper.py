from typing import Final, Self, Set

import cv2 as cv
import gym
import numpy as np

__all__ = ["PongWrapper"]


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

    def __init__(self: Self, env_name: str):
        env = gym.make(env_name)
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def step(self: Self, action: int):
        """Map the reduced action space to the original actions."""
        action = self.default_action if action not in self.allowed_actions else action
        next_state, reward, done, _, _ = super().step(action)
        return preprocess_state(next_state), reward, done

    def reset(self: Self):
        return preprocess_state(self.env.reset()[0])
