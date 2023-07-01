from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn

from ._dqn_base_agent import DqnBaseAgent


class DuelingDQNAgent(DqnBaseAgent):
    """A dueling deep-Q-network agent."""

    @classmethod
    @property
    def name(cls) -> str:
        return "dueling_dqn"

    def __init__(
        self: Self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__extend_model()

    def __extend_model(self: Self) -> None:
        """Splits model into advantage and value streams."""

        # remove last layer of self.model
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device_)

        output_dim = self.model[-1].out_features
        self.advantage = nn.Sequential(
            nn.Linear(output_dim, 256),  # type:ignore
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )
        self.advantage.to(self.device_)
        self.value = nn.Sequential(
            nn.Linear(output_dim, 256),  # type:ignore
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value.to(self.device_)

    @torch.no_grad()
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        return self.forward(next_states).max(1)[0].unsqueeze(1)

    def forward(self: Self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type:ignore
            feature = self.model(x)
            advantage = self.advantage(feature)
            value = self.value(feature)
            return value + advantage - advantage.mean()

    def load(self: Self, name: Path) -> None:
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint["model"])
        self.advantage.load_state_dict(checkpoint["advantage"])
        self.value.load_state_dict(checkpoint["value"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])

    def save(self: Self, name: Path) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "advantage": self.advantage.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(checkpoint, name)
