from ._base_env import BaseEnvWrapper
from .pong_env import PongEnvWrapper

env_registry = [PongEnvWrapper]

__all__ = ["BaseEnvWrapper", "env_registry"]
