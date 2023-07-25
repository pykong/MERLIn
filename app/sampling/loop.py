import logging
import logging.handlers
import multiprocessing
import pickle
import random
import time
import zlib
from pathlib import Path
from queue import Empty
from typing import Self

from app.loop import make_env
from app.memory.transition import Transition
from app.sampling.config import SamplingConfig


class TransitionDump:
    def __init__(self: Self, capacity: int, name: str) -> None:
        self.capacity = capacity
        manager = multiprocessing.Manager()
        self.buffer = manager.list()  # shared list
        self.name = name  # new attribute to name the file
        self.len = multiprocessing.Value("i", 0)  # shared integer for length tracking

    def push(self: Self, transition: Transition) -> None:
        if not self.is_full():
            bytes_ = zlib.compress(pickle.dumps(transition))
            self.buffer.append(bytes_)
            self.len.value += 1

    def __len__(self: Self) -> int:
        return self.len.value

    def is_full(self: Self) -> bool:
        return len(self) >= self.capacity

    def save_buffer(self: Self) -> None:  # new method to save the buffer to a file
        path = Path(f"{self.name}.pkl")
        with path.open("wb") as f:
            pickle.dump(self.buffer[:], f)  # write the buffer to a file


def loop(config: SamplingConfig, dumps: tuple) -> None:
    # create environment
    env = make_env(
        config.env_name,
        state_dims=(config.input_dim, config.input_dim),
        skip=config.frame_skip,
        step_penalty=config.step_penalty,
        stack_size=config.num_stacked_frames,
    )

    # start environment
    state = env.reset()
    done = False

    step = 0
    while not all([d.is_full() for d in dumps]):
        if done:
            state = env.reset()

        # feign action
        action = random.randrange(0, env.action_space.n)  # type:ignore
        next_state, reward, done = env.step(action)

        # save experience
        transition = Transition(state, action, reward, next_state, done)
        dumps[0].push(transition)
        if reward < 0.0:
            dumps[1].push(transition)
        if reward > 0.0:
            dumps[2].push(transition)


def run_multiprocess_loop(config: SamplingConfig):
    # init dumps
    unbiased_dump = TransitionDump(config.sample_count, "unbiased_dump")
    negative_dump = TransitionDump(config.sample_count, "negative_dump")
    positive_dump = TransitionDump(config.sample_count, "positive_dump")
    dumps = (
        unbiased_dump,
        negative_dump,
        positive_dump,
    )

    # create processes
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=loop, args=(config, dumps))
        processes.append(p)

    # start processes
    for p in processes:
        p.start()

    # while processes are running, print the size of the dumps
    while any(p.is_alive() for p in processes):
        print(f"Unbiased dump size: {len(unbiased_dump)}")
        print(f"Negative dump size: {len(negative_dump)}")
        print(f"Positive dump size: {len(positive_dump)}")
        time.sleep(1)  # sleep for a while to not overload the console output

    # wait for all processes to finish
    for p in processes:
        p.join()

    # after all processes are done, save the buffers
    for dump in dumps:
        dump.save_buffer()
