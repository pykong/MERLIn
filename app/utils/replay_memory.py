import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(zip(*experience))

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, sample_size)
        return batch

    def __len__(self):
        return len(self.buffer)
