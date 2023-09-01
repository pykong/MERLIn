from app.nets._base_net import BaseNet
from app.nets.conv_net import ConvNet
from app.nets.linear_deep_net import LinearDeepNet
from app.nets.linear_flat_net import LinearFlatNet

net_registry = [
    LinearFlatNet,
    LinearDeepNet,
    ConvNet,
]


def make_net(name: str) -> BaseNet:
    """Create neural net of provided name.

    Args:
        name (str): The identifier string of the neural network.

    Returns:
        BaseNet: The neural network instance.
    """
    net = [net for net in net_registry if net.name == name][0]
    return net()


__all__ = ["BaseNet", "make_net"]
