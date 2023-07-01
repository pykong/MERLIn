from ._base_agent import BaseAgent
from .dqn_double import DoubleDQNAgent
from .dqn_double_duelling import DoubleDuelingDQNAgent
from .dqn_dueling import DuelingDQNAgent
from .dqn_vanilla import VanillaDQNAgent

agent_registry = [
    DoubleDQNAgent,
    DoubleDuelingDQNAgent,
    DuelingDQNAgent,
    VanillaDQNAgent,
]

__all__ = ["BaseAgent", "agent_registry"]
