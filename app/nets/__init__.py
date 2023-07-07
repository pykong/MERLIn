from ._base_net import BaseNet
from .linear_flat_net import LinearFlatNet
from .linear_deep_net import LinearDeepNet
from .conv_net import ConvNet

net_registry = [
    LinearFlatNet,
    LinearDeepNet,
    ConvNet,
]

__all__ = ["BaseNet", "net_registry"]
