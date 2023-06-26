from ._base_agent import BaseAgent
from .dqn_double import DoubleDQNAgent
from .dqn_double_duelling import DoubleDuellingDQNAgent
from .dqn_duelling import DuellingDQNAgent
from .dqn_vanilla import VanillaDQNAgent

agent_registry = [
    DoubleDQNAgent,
    DoubleDuellingDQNAgent,
    DuellingDQNAgent,
    VanillaDQNAgent,
]

__all__ = ["BaseAgent", "agent_registry"]
