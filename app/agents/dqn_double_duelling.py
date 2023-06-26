from typing import Final

from .dqn_double import DoubleDQNAgent
from .dqn_duelling import DuellingDQNAgent


class DoubleDuellingDQNAgent(DoubleDQNAgent, DuellingDQNAgent):
    """A chimeric agent incoorporating a target and duellling network."""

    name: Final[str] = "double_duelling_dqn"

    # TODO: Ensure correct MRO: _calc_max_q_prime() from DoubleDQNAgent, forward() from DuellingDQNAgent
