import random
from collections import deque
from typing import NamedTuple, Self

from numpy import ndarray


class Experience(NamedTuple):
    state: ndarray
    action: int
    reward: float
    next_state: ndarray
    done: bool


class ReplayMemory:
    def __init__(self: Self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self: Self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self: Self, batch_size: int) -> list[Experience]:
        sample_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, sample_size)
        return batch

    def __len__(self):
        return len(self.buffer)
