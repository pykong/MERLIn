from ._base_agent import BaseAgent
from .dqn_double import DoubleDQNAgent
from .dqn_duelling import DuellingDQNAgent
from .dqn_vanilla import VanillaDQNAgent

agent_registry = [DoubleDQNAgent, DuellingDQNAgent, VanillaDQNAgent]

__all__ = ["BaseAgent", "agent_registry"]
