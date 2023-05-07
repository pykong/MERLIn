from typing import Final, Self, Set

import gym


class PongWrapper(gym.Wrapper):
    # https://gymnasium.farama.org/environments/atari/pong/#actions
    default_action: Final[int] = 0
    allowed_actions: Final[Set[int]] = {0, 1, 2, 3}  # TODO is '1' needed?

    def __init__(self: Self, env_name: str):
        env = gym.make(env_name)
        super(PongWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def step(self: Self, action: int):
        """Map the reduced action space to the original actions."""
        action = self.default_action if action not in self.allowed_actions else action
        return super(PongWrapper, self).step(action)
