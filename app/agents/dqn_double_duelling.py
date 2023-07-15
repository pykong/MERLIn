from app.agents.dqn_double import DoubleDQNAgent
from app.agents.dqn_dueling import DuelingDQNAgent


class DoubleDuelingDQNAgent(DoubleDQNAgent, DuelingDQNAgent):
    """A chimeric agent combining aspects from double and dueling DQN learning."""

    @classmethod
    @property
    def name(cls) -> str:
        return "double_dueling_dqn"
