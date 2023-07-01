from ._dqn_base_agent import DqnBaseAgent
from .dqn_double import DoubleDQNAgent
from .dqn_double_duelling import DoubleDuelingDQNAgent
from .dqn_dueling import DuelingDQNAgent
from .dqn_vanilla import VanillaDQNAgent
from .random_walker import RandomWalkerAgent

agent_registry = [
    DoubleDQNAgent,
    DoubleDuelingDQNAgent,
    DuelingDQNAgent,
    VanillaDQNAgent,
    RandomWalkerAgent,
]

__all__ = ["DqnBaseAgent", "agent_registry"]
