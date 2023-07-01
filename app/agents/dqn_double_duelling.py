from .dqn_double import DoubleDQNAgent
from .dqn_dueling import DuelingDQNAgent


class DoubleDuelingDQNAgent(DoubleDQNAgent, DuelingDQNAgent):
    """A chimeric agent combining aspects from double and dueling DQN learning."""

    @classmethod
    @property
    def name(cls) -> str:
        return "double_dueling_dqn"
