from app.agents._dqn_base_agent import DqnBaseAgent
from app.agents.dqn_double import DoubleDQNAgent
from app.agents.dqn_double_duelling import DoubleDuelingDQNAgent
from app.agents.dqn_dueling import DuelingDQNAgent
from app.agents.dqn_vanilla import VanillaDQNAgent
from app.agents.random_walker import RandomWalkerAgent

agent_registry = [
    DoubleDQNAgent,
    DoubleDuelingDQNAgent,
    DuelingDQNAgent,
    VanillaDQNAgent,
    RandomWalkerAgent,
]

__all__ = ["DqnBaseAgent", "agent_registry"]
