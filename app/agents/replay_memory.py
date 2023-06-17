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
    def __init__(self: Self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self: Self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self: Self) -> list[Transition]:
        sample_size = min(len(self.buffer), self.batch_size)
        indices = np.random.choice(len(self.buffer), sample_size - 1, replace=False)
        batch = [self.buffer[i] for i in indices]
        batch.append(self.buffer[-1])
        return batch

    def __len__(self: Self) -> int:
        return len(self.buffer)
