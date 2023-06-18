import numpy as np

from ._base_env import BaseEnvWrapper
from .action import Action


class PongEnvWrapper(BaseEnvWrapper):
    @classmethod
    @property
    def name(cls) -> str:
        """String to represent wrapper to the outside."""
        return "pong"

    @classmethod
    @property
    def env_name(cls) -> str:
        """String to represent wrapper to the outside."""
        return "ALE/Pong-v5"

    @classmethod
    @property
    def valid_actions(cls) -> set[int]:
        """Set of valid actions to chose."""
        return {Action.NOOP, Action.RIGHT, Action.LEFT}

    @classmethod
    def _crop_state(cls, state: np.ndarray) -> np.ndarray:
        """Crop state to informative region."""
        return state[33:194, 16:-16]
