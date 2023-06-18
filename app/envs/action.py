from enum import UNIQUE, IntEnum, verify

__all__ = ["Action"]


@verify(UNIQUE)
class Action(IntEnum):
    """Actions of Atari game environments.

    https://gymnasium.farama.org/environments/atari/pong/#actions
    """

    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5
