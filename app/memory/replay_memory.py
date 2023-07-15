import pickle
import zlib
from collections import deque
from typing import Deque, Self

import numpy as np

from app.memory.transition import Transition


def ensure_transitions(func):
    """Ensure buffer has at least one transition, else raise ValueError."""

    def _decorator(self, *args, **kwargs):
        if len(self) == 0:
            raise ValueError("Attempt to sample empty replay memory.")
        return func(self, *args, **kwargs)

    return _decorator


class ReplayMemory:
    def __init__(self: Self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self: Self, transition: Transition) -> None:
        self.buffer.append(zlib.compress(pickle.dumps(transition)))

    @ensure_transitions
    def __draw_random_indices(self: Self) -> list[int]:
        """Draw random indices, always include most recent transition."""
        sample_size = min(len(self.buffer), self.batch_size) - 1
        indices = np.random.choice(len(self), sample_size, replace=False).tolist()
        indices.append(-1)
        return indices

    @ensure_transitions
    def __pad_indices(self: Self, batch: list[int]) -> list[int]:
        """Pad batch if it is smaller than configured size."""
        pad = [-1] * (self.batch_size - len(batch))
        batch.extend(pad)
        return batch

    @ensure_transitions
    def sample(self: Self) -> list[Transition]:
        """Sample batch of pre-configured size."""
        indices = self.__draw_random_indices()
        indices = self.__pad_indices(indices)
        batch = [pickle.loads(zlib.decompress(self.buffer[i])) for i in indices]
        return batch

    def __len__(self: Self) -> int:
        return len(self.buffer)
