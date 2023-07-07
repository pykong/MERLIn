from ._base_net import BaseNet
from .linear_flat_net import LinearFlatNet
from .linear_deep_net import LinearDeepNet
from .nature_net import NatureNet

net_registry = [
    LinearFlatNet,
    LinearDeepNet,
    NatureNet,
]

__all__ = ["BaseNet", "net_registry"]
