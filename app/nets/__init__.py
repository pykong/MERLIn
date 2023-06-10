from ._base_net import BaseNet
from .ben_net import BenNet

net_registry = [BenNet]

__all__ = ["BaseNet", "net_registry"]
