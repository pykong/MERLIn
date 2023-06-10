from ._base_net import BaseNet
from .ben_net import BenNet
from .nature_net import NatureNet
from .neuroips_net import NeuroipsNet

net_registry = [BenNet, NatureNet, NeuroipsNet]

__all__ = ["BaseNet", "net_registry"]
