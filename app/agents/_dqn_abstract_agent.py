import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, Optional, Self

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn

from app.memory import ReplayMemory, Transition
from app.nets import BaseNet
from app.utils.logging import LogLevel, logger


class Minibatch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


def get_torch_device() -> torch.device:
    """Provide best possible device for running PyTorch."""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda  # type: ignore
        logger.log(str(LogLevel.GREEN), f"CUDA is available (v{cuda_version}).")
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_name(i)
            logger.log(str(LogLevel.GREEN), f"cuda:{i} - {gpu}")
        torch.backends.cudnn.benchmark = True  # type: ignore
        return torch.device("cuda")
    else:
        logger.log(str(LogLevel.YELLOW), "Running PyTorch on CPU.")
        return torch.device("cpu")


class DqnAbstractAgent(ABC, pl.LightningModule):
    def __init__(
        self: Self,
        state_shape: tuple[int, int, int],
        action_space: int,
        net: BaseNet,
        alpha: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        gamma: float = 0.99,
        memory_size: int = 10_000,
        batch_size: int = 64,
        use_amp: bool = False,
        spice_memory: bool = False,
        **kwargs: Optional[Any],
    ):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = action_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.use_amp = use_amp
        self.spice_memory = spice_memory
        self.memory = ReplayMemory(capacity=memory_size, batch_size=batch_size)
        self.device_: torch.device = get_torch_device()
        self.model = net.build_net(self.state_shape, self.num_actions, self.device_)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=alpha)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)  # type:ignore

    def replay(self: Self) -> float:
        # sample memory
        sample = self.memory.sample()

        # convert the minibatch to a more convenient format
        states, actions, rewards, next_states, dones = self._encode_minibatch(sample)

        # mask dones
        dones = 1 - dones

        # predict Q-values for the initial states
        q_out = self.forward(states)

        # state_action_values
        q_a = q_out.gather(1, actions)

        # calc max q prime value
        max_q_prime = self._calc_max_q_prime(next_states)

        # compute the expected Q values (expected_state_action_values)
        target = rewards + self.gamma * max_q_prime * dones

        # spice-up memory
        if self.spice_memory:
            max_surprise = self.find_max_surprise(q_a, target, sample)
            self.memory.push(max_surprise)

        # calc losses
        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type:ignore
            losses = F.smooth_l1_loss(q_a, target)

        # update the weights
        self._update_weights(losses)

        # return losses
        return losses.mean().item()

    @abstractmethod
    def _calc_max_q_prime(self: Self, next_states: Tensor) -> float:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    def find_max_surprise(
        self: Self, q_a: Tensor, target: Tensor, sample: list[Transition]
    ) -> Transition:
        """Return the transition with the highest surprise (absolute TD error)."""
        abs_td_errors = torch.abs(target - q_a).detach().cpu().numpy()
        max_surprise_idx = abs_td_errors.argmax()
        return sample[max_surprise_idx]

    def _encode_minibatch(self: Self, transitions: list[Transition]) -> Minibatch:
        def encode_array(states: list[np.ndarray]) -> Tensor:
            return torch.from_numpy(np.array(states)).to(self.device_).float()

        def encode_number(number: list[float] | list[int]) -> Tensor:
            return torch.tensor(number, device=self.device_).unsqueeze(-1)

        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        rewards = [t.reward for t in transitions]
        next_states = [t.next_state for t in transitions]
        dones = [float(t.done) for t in transitions]

        return Minibatch(
            states=encode_array(states),
            actions=encode_number(actions),
            rewards=encode_number(rewards),
            next_states=encode_array(next_states),
            dones=encode_number(dones),
        )

    def _update_weights(self: Self, losses: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(losses).backward()  # type: ignore
        # Unscales the gradients of optimizer's assigned params in-place
        # https://h-huang.github.io/tutorials/recipes/recipes/amp_recipe.html#inspecting-modifying-gradients-e-g-clipping
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # type: ignore
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def remember(self: Self, transition: Transition) -> None:
        self.memory.push(transition)

    @torch.no_grad()
    def act(self: Self, state: np.ndarray) -> int:
        """Take random action with probability epsilon, else take best action."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.from_numpy(state).to(self.device_)
        act_values = self.forward(state.unsqueeze(0))
        return act_values.argmax().item()

    def forward(self: Self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type:ignore
            return self.model(x)  # type:ignore

    def update_epsilon(self: Self, epsilon_step: float) -> None:
        """Decrease epsilon linearly by epsilon step.

        Args:
            epsilon_step (float): The amount to subtract from epsilon.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= epsilon_step

    def load(self: Self, name: Path) -> None:
        """Load model from path.

        Args:
            name (Path): The path to the model file.
        """
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])

    def save(self: Self, name: Path) -> None:
        """Save model to path.

        Args:
            name (Path): The file path to save the model to.
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(checkpoint, name)
