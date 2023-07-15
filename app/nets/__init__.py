from app.nets._base_net import BaseNet
from app.nets.linear_flat_net import LinearFlatNet
from app.nets.linear_deep_net import LinearDeepNet
from app.nets.conv_net import ConvNet

net_registry = [
    LinearFlatNet,
    LinearDeepNet,
    ConvNet,
]

__all__ = ["BaseNet", "net_registry"]
