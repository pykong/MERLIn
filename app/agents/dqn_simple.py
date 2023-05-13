import random
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class DQNSimpleAgent:
    def __init__(
        self,
        state_shape,
        action_space,
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
    ):
        self.state_shape = state_shape
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.Experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.action_space, activation="linear")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def act(self, state) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state)
        return int(np.argmax(act_values[0]))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = np.expand_dims(state, axis=0)  # add batch dimension
            next_state = np.expand_dims(next_state, axis=0)  # add batch dimension
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
