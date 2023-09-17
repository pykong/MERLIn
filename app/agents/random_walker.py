import random
from pathlib import Path
from typing import Self

from torch import Tensor

from app.agents._dqn_abstract_agent import DqnAbstractAgent
from app.memory import Transition


class RandomWalkerAgent(DqnAbstractAgent):
    """An agent taking random actions, not learning."""

    @classmethod
    @property
    def name(cls) -> str:
        return "random_walker"

    def __init__(
        self: Self,
        *args,
        **kwargs,
    ):
        self.num_actions = kwargs["action_space"]
        self.epsilon = 0.0
        self.epsilon_min = 0.0

    def act(self: Self, state) -> int:
        return random.randrange(self.num_actions)

    def replay(self: Self) -> float:
        return 0.0

    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        return 0.0

    def remember(self: Self, transition: Transition) -> None:
        pass

    def load(self: Self, name: Path) -> None:
        pass

    def save(self: Self, name: Path) -> None:
        pass
