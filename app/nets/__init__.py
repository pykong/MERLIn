from ._base_net import BaseNet
from .ben_net import BenNet
from .nature_net import NatureNet

net_registry = [BenNet, NatureNet]

__all__ = ["BaseNet", "net_registry"]
