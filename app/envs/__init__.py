from app.envs._base_env import BaseEnvWrapper
from app.envs.pong_env import PongEnvWrapper

env_registry = [PongEnvWrapper]

__all__ = ["BaseEnvWrapper", "env_registry"]
