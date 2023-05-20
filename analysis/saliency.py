import matplotlib.pyplot as plt
import numpy as np
import torch

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
plt.axis('off')
plt.show()
