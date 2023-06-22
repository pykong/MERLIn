from ._base_net import BaseNet
from .ben_net import BenNet
from .big_net import BigNet
from .linear_big_net import LinearBigNet
from .linear_net import LinearNet
from .nature_net import NatureNet
from .neuroips_net import NeuroipsNet

net_registry = [BenNet, BigNet, LinearBigNet, LinearNet, NatureNet, NeuroipsNet]

__all__ = ["BaseNet", "net_registry"]
