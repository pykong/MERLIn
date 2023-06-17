from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseNet(ABC):
    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    def build_net(
        self,
        state_shape: tuple[int, int, int],
        num_actions: int,
        device: torch.device,
        use_amp: bool = False,
    ) -> nn.Sequential:
        model = self._define_net(state_shape, num_actions)
        model = model.to(device=device)
        if use_amp:
            model = model.to(memory_format=torch.channels_last)  # type:ignore
        return model

    @abstractmethod
    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
        raise NotImplementedError()

    @staticmethod
    def _calc_conv_outdim(dim: int, kernel_size: int, stride: int, padding: int) -> int:
        """Calculate the size of the output of a conv layer."""
        return ((dim + 2 * padding - kernel_size) // stride) + 1
