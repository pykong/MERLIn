from collections import deque
from typing import Deque, Self

import numpy as np

from .transition import Transition


class ReplayMemory:
    def __init__(self: Self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self: Self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self: Self) -> list[Transition]:
        if len(self.buffer) == 0:
            raise ValueError("Attempt to sample empty replay memory.")
        sample_size = min(len(self.buffer), self.batch_size) - 1
        indices = np.random.choice(len(self), sample_size, replace=False).tolist()
        indices.append(-1)
        
        batch = [self.buffer[i] for i in indices]

        # ensure correct batch size via padding
        pad = [self.buffer[-1]] * (self.batch_size - len(batch))
        batch.extend(pad)

        return batch

    def __len__(self: Self) -> int:
        return len(self.buffer)
