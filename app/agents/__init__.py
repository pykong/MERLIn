from ._base_agent import BaseAgent
from .dqn_double import DoubleDQNAgent
from .dqn_vanilla import VanillaDQNAgent

agent_registry = [
    DoubleDQNAgent,
    VanillaDQNAgent,
]

__all__ = ["BaseAgent", "agent_registry"]
