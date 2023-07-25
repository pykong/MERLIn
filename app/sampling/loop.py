import pickle
import random
import zlib
from pathlib import Path
from typing import Self

from app.loop import make_env  # TODO: Relocate factory funcs
from app.memory.transition import Transition
from app.sampling.config import SamplingConfig


class TransitionDump:
    def __init__(self: Self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[bytes] = []

    def push(self: Self, transition: Transition) -> None:
        if not self.is_full():
            bytes_ = zlib.compress(pickle.dumps(transition))
            self.buffer.append(bytes_)

    def __len__(self: Self) -> int:
        return len(self.buffer)

    def is_full(self: Self) -> bool:
        return len(self) >= self.capacity


def loop(config: SamplingConfig) -> None:
    # init dumps
    unbiased_dump = TransitionDump(config.sample_count)
    negative_dump = TransitionDump(config.sample_count)
    positive_dump = TransitionDump(config.sample_count)
    dumps = (
        unbiased_dump,
        negative_dump,
        positive_dump,
    )

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
        unbiased_dump.push(transition)
        if reward < 0.0:
            negative_dump.push(transition)
        if reward > 0.0:
            positive_dump.push(transition)

        # provide output
        print(
            " ".join(
                (
                    f"S:{step:08d} :",
                    f"U{len(unbiased_dump):06d}",
                    f"| -{len(negative_dump):06d}",
                    f"| +{len(positive_dump):06d}",
                )
            )
        )

        # update state
        state = next_state

        # increment step counter
        step += 1
