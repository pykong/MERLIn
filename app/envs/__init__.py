from typing import Any

from app.envs._base_env import BaseEnvWrapper
from app.envs.pong_env import PongEnvWrapper

env_registry = [PongEnvWrapper]


def make_env(name: str, **kwargs: Any) -> BaseEnvWrapper:
    """Create environment wrapper of provided name.

    Args:
        name (str): The identifier string of the environment wrapper.

    Returns:
        BaseEnvWrapper: A wrapper instance of the environment.
    """
    env_ = [e for e in env_registry if e.name == name][0]
    return env_(**kwargs)


__all__ = ["BaseEnvWrapper", "make_env"]
