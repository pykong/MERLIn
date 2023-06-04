from collections import deque
from typing import Deque, NamedTuple, Self

import numpy as np
from numpy import ndarray


class Transition(NamedTuple):
    state: ndarray
    action: int
    reward: float
    next_state: ndarray
    done: bool


class ReplayMemory:
    def __init__(self: Self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self: Self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self: Self, batch_size: int) -> list[Transition]:
        sample_size = min(len(self.buffer), batch_size)
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self: Self) -> int:
        return len(self.buffer)
