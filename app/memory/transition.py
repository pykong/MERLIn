from typing import NamedTuple

from numpy import ndarray


class Transition(NamedTuple):
    state: ndarray
    action: int
    reward: float
    next_state: ndarray
    done: bool
