import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ..app.nets.ben_net import BenNet


def plot_saliency_map(dqn_model: nn.Sequential, state) -> None:
    # Assuming that you have a trained DQN model 'dqn_model' and an input state 'state'
    # Ensure that the model is in evaluation mode
    dqn_model.eval()

    # Also assuming 'state' is a torch tensor
    state.requires_grad = True

    # Forward pass
    q_values = dqn_model(state)

    # Take the maximum Q-value
    q_values_max = q_values.max()

    # Backward pass
    q_values_max.backward()

    # Calculate saliency map: take absolute value of gradients and then take the maximum across color channels
    saliency, _ = torch.max(state.grad.data.abs(), dim=1)

    # Detach the saliency map from the computational graph and convert to numpy array
    saliency = saliency.detach().numpy()

    # Plot saliency map
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.axis("off")
    plt.savefig("out/saliency_map.svg")
    plt.show()


if __name__ == "__main__":
    model = BenNet().build_net(None, None, None)
    with open("xxx", "r") as cp_f:
        model.load_state_dict(torch.load(cp_f))
    state = None
    plot_saliency_map(model, state)
