from typing import Any

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


def make_agent(agent_name: str, **kwargs: Any) -> DqnBaseAgent:
    """Create agent of provided name and inject neural network.

    Args:
        agent_name (str): The identifier string of the agent.

    Returns:
        DqnBaseAgent: The agent instance.
    """
    agent_ = [a for a in agent_registry if a.name == agent_name][0]
    return agent_(**kwargs)


__all__ = ["DqnBaseAgent", "make_agent"]
