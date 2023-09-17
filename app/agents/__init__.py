from typing import Any

from app.agents._dqn_abstract_agent import DqnAbstractAgent
from app.agents.dqn_basic import BasicDQNAgent
from app.agents.dqn_double import DoubleDQNAgent
from app.agents.dqn_dueling import DuelingDQNAgent
from app.agents.random_walker import RandomWalkerAgent

agent_registry = [
    BasicDQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    RandomWalkerAgent,
]


def make_agent(agent_name: str, **kwargs: Any) -> DqnAbstractAgent:
    """Create agent of provided name and inject neural network.

    Args:
        agent_name (str): The identifier string of the agent.

    Returns:
        DqnAbstractAgent: The agent instance.
    """
    agent_ = [a for a in agent_registry if a.name == agent_name][0]
    return agent_(**kwargs)


__all__ = ["DqnAbstractAgent", "make_agent"]
