import os
import random
from collections import deque

import cv2
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
MODEL_SAVE_INTERVAL = 1000
LOG_INTERVAL = 10


class DQN:
    def __init__(self, input_shape, num_actions):
        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (8, 8),
                    strides=(4, 4),
                    activation="relu",
                    input_shape=input_shape,
                ),
                Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
                Conv2D(64, (3, 3), activation="relu"),
                Flatten(),
                Dense(512, activation="relu"),
                Dense(num_actions),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse"
        )

    def update(self, states, actions, rewards, next_states, dones):
        target_values = self.model.predict(states)
        next_target_values = self.model.predict(next_states)
        for i in range(states.shape[0]):
            target_values[i, actions[i]] = rewards[i] + GAMMA * np.max(
                next_target_values[i]
            ) * (1 - dones[i])
        self.model.train_on_batch(states, target_values)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.output_shape[1])
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0])


def preprocess_state(state):
    state = state[35:195]  # Crop the irrelevant parts of the image (top and bottom)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    state = cv2.resize(
        state, (80, 80), interpolation=cv2.INTER_AREA
    )  # Downsample to 80x80
    state = state[
        :, :, np.newaxis
    ]  # Add a channel dimension for the neural network input
    return state


def main():
    env = gym.make("PongDeterministic-v4")
    input_shape = (80, 80, 1)
    num_actions = env.action_space.n

    dqn = DQN(input_shape, num_actions)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    win_count = 0

    with open("training_metrics.log", "w") as log_file:
        log_file.write("episode,avg_reward,win_rate,avg_episode_length,epsilon\n")

        episode = 0
        while True:
            episode += 1
            state = preprocess_state(env.reset()[0])
            done = False

            episode_reward = 0
            episode_length = 0

            while not done:
                action = dqn.act(state, epsilon)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = preprocess_state(next_state)
                memory.append((state, action, reward, next_state, done))
                state = next_state

                episode_reward += reward
                episode_length += 1

                if len(memory) >= BATCH_SIZE:
                    minibatch = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = map(
                        np.array, zip(*minibatch)
                    )
                    dqn.update(states, actions, rewards, next_states, dones)

                total_steps += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if episode_reward > 0:
                    win_count += 1
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

                if episode % LOG_INTERVAL == 0:
                    avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
                    avg_episode_length = np.mean(episode_lengths[-LOG_INTERVAL:])
                    win_rate = (win_count / LOG_INTERVAL) * 100

                    log_message = f"{episode},{avg_reward:.2f},{win_rate:.2f},{avg_episode_length:.2f},{epsilon:.4f}"
                    print(log_message)
                    log_file.write(log_message + "\n")
                    win_count = 0

                if total_steps % MODEL_SAVE_INTERVAL == 0:
                    dqn.model.save(f"pong_model_{total_steps}.h5")


if __name__ == "__main__":
    main()
