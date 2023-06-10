from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseNet(ABC):
    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @abstractmethod
    def build_net(
        self, state_shape: tuple[int, int, int], num_actions: int, device: torch.device
    ) -> nn.Sequential:
        raise NotImplementedError()

    @staticmethod
    def _calc_conv_outdim(dim: int, kernel_size: int, stride: int, padding: int) -> int:
        """Calculate the size of the output of a conv layer."""
        return ((dim + 2 * padding - kernel_size) // stride) + 1
