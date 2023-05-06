import cv2
import gym
import numpy as np


def preprocess_state(state):
    state = state[0][35:195]  # Crop the irrelevant parts of the image (top and bottom)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(
        state, (80, 80), interpolation=cv2.INTER_AREA
    )  # Downsample to 80x80
    state = state[
        :, :, np.newaxis
    ]  # Add a channel dimension for the neural network input
    return state


env = gym.make("PongDeterministic-v4")
state = env.reset()

print("Original state shape:", state[0].shape)

processed_state = preprocess_state(state)
print("Processed state shape:", processed_state.shape)
