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
        self.buffer: Deque[bytes] = deque(maxlen=capacity)

    def push(self: Self, transition: Transition) -> None:
        bytes_ = zlib.compress(pickle.dumps(transition))
        self.buffer.append(bytes_)

    def __getitem__(self: Self, index: int) -> Transition:
        bytes_ = self.buffer[index]
        return pickle.loads(zlib.decompress(bytes_))

    def __len__(self: Self) -> int:
        return len(self.buffer)

    @ensure_transitions
    def __draw_random_indices(self: Self) -> list[int]:
        """Draw random indices of transition entries.

        Always include most recent transition (combined experience replay).
        Pad if the current memory size is smaller than the configured batch size.

        Returns:
            list[int]: The drawn indcies.
        """
        sample_size = min(len(self), self.batch_size) - 1
        indices = np.random.choice(len(self), sample_size, replace=False).tolist()
        pad = [-1] * (self.batch_size - len(indices))
        return [*indices, *pad]

    @ensure_transitions
    def sample(self: Self) -> list[Transition]:
        """Sample batch of pre-configured size.

        Returns:
            list[Transition]: The sampled transitions.
        """
        indices = self.__draw_random_indices()
        return [self[i] for i in indices]
