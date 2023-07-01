from ._base_net import BaseNet
from .linear_mini_net import LinearMiniNet
from .linear_net import LinearNet
from .linear_net_bn import LinearNetBN
from .nature_net import NatureNet

net_registry = [
    LinearMiniNet,
    LinearNet,
    LinearNetBN,
    NatureNet,
]

__all__ = ["BaseNet", "net_registry"]
