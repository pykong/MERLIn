import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from app.utils.replay_memory import Experience


class DQN:
    def __init__(self, state_shape, action_space, gamma=0.99, alpha=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.model = self.__build_model()

    def __build_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (8, 8),
                strides=(4, 4),
                activation="relu",
                input_shape=self.state_shape,
            )
        )
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.action_space))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(lr=self.alpha))
        return model

    def act(self, state, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # random exploration
            return np.random.randint(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def update(self, experience: Experience):
        

        # Predict Q(s, a) given the batch of states
        q_sa = self.model.predict(states)

        # Predict Q(s', a') from the evaluation network
        q_sa_next = self.model.predict(next_states)

        # Predict Q(s', a') from the target network
        q_sa_next_target = target_model.predict(next_states)

        # Update target values
        targets = q_sa
        batch_size = states.shape[0]
        for i in range(batch_size):
            # correction on the Q value for the action used
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                targets[i][actions[i]] = (
                    rewards[i]
                    + self.gamma * q_sa_next_target[i][np.argmax(q_sa_next[i])]
                )

        # Train the neural network
        self.model.train_on_batch(states, targets)

    def save_model(self, file_path):
        self.model.save(file_path)

    def copy_from(self, other_agent: "DQN"):
        self.model.set_weights(other_agent.model.get_weights())
