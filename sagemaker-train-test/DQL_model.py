"""
DQL 8 was perfect, but memory lekage ocured, here memory lekage is adressed and the code is getting refactured

same as 7 only that it trains on entire batches

"""

import numpy as np
import random

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model

from collections import deque


def tensorboard_log_scalar(writer, tag, value, step):
    """
    Log a scalar value to both tensorboard and stdout
    """
    writer.add_scalar(tag, value, step)
    print(f'{tag}: {value}')


class DQLModel:
    """
    This class implements an agent for trading in the stock market, which uses a deep Q-network (DQN)
    to make trading decisions. The agent maintains a memory of its past experiences,
    which consists of tuples of (state, action, reward, next_state, done).
    The agent uses this memory to train the DQN, where it learns to approximate the Q-function,
    which represents the expected return of performing an action in a certain state.

    The agent uses two networks, a main network and a target network, to stabilize the training process.
    The main network is updated in each training step, while the target network is updated less frequently
    (every updateTargetEvery steps) with the weights of the main network.
    The agent also uses an epsilon-greedy strategy for action selection,
    where it starts with a high exploration rate (epsilon) and decays it over time.
    """

    # Constructor function
    def __init__(self, state_size, action_space=2, model_name="AITrader_v1", update_target_model_every=5,
                 loadWeights="", useCustomModel=""):
        # Initialize class variables
        self.state_size = state_size  # The size of the state space
        self.action_space = action_space  # The number of possible actions
        self.memory = deque(maxlen=100_000)  # Memory of the agent to store past experiences
        self.inventory = []  # List to store the trades made by the agent
        self.model_name = model_name  # Name of the model

        self.reshapeDimForPrediction = (1,) + state_size  # Shape of the input for model prediction

        # Parameters for Q-learning
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_final = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration

        # Parameters for updating the target network
        self.updateTargetEvery = 20
        self.targetUpdateCounter = 0

        # TensorBoard for visualization
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(self.model_name))

        # Load the model or build a new one
        if loadWeights != "":
            self.model = load_model(loadWeights)
        elif useCustomModel != "":
            self.model = useCustomModel
        else:
            self.model = self._model_builder_v1()

        self.model.summary()

        # Initialize the target network with the same weights as the main network
        self.target_model = self.model
        self.target_model.set_weights(self.model.get_weights())

    def _model_builder_v1(self):
        # Builds and returns a new model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        model.add(tf.keras.layers.Reshape((32 * self.state_size[0],)))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def trade(self, state):
        # Choose an action based on the current state
        if random.random() <= self.epsilon:
            # Choose a random action
            return random.randrange(self.action_space)
        else:
            # Choose the best action predicted by the model
            actions = self.model.predict(state.reshape(self.reshapeDimForPrediction))
            return np.argmax(actions[0])

    def batch_train(self, batchSize, samplingMethod="last_steps"):
        # Train the model on a batch of experiences
        if samplingMethod == "last_steps":
            batch = list(self.memory)[-batchSize:]
        else:
            batch = random.sample(self.memory, batchSize)

        # Separate the states and the future states from the batch
        currentStates = np.array([transition[0] for transition in batch])
        newCurrentStates = np.array([transition[3] for transition in batch])

        # Predict the Q-values of the current states and the future states
        currentQsList = self.model.predict(currentStates)
        futureQsList = self.target_model.predict(newCurrentStates)

        X = []
        y = []

        # Update the Q-values based on the rewards received and the future Q-values
        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                newQ = reward + self.gamma * np.amax(futureQsList[index])

            currentQs = currentQsList[index]
            currentQs[action] = newQ

            X.append(state)
            y.append(currentQs)

        # Train the model on the states and the updated Q-values
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0, callbacks=[self.tensorboard])

        # Update the target network if necessary
        self.targetUpdateCounter += 1
        if self.targetUpdateCounter > self.updateTargetEvery:
            self.target_model.set_weights(self.model.get_weights())
            self.targetUpdateCounter = 0

        # Decay the exploration rate
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay





