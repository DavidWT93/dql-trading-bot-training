from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class TradingEnvironment:
    """
    This class creates a trading environment for a stock. The class is initialized with three main attributes:
    the input data for the machine learning model, the size of the sliding window for creating states, and the closing
    prices of the stock. Each state is a sliding window of the input data, which is normalized between 0 and 1.
    The environment calculates the rewards for the states based on the percentage change in closing price.
    There is a reward for both buying and selling actions. The environment also allows to add an extra dimension
    to a state to represent how long a stock has been held.

    """
    def __init__(self, inputData, windowSize, closePrice):
        self.inputData = inputData  # The input data for the model
        self.windowSize = windowSize  # The size of the sliding window for creating states
        self.closePrice = closePrice  # The closing prices of the stock

        self._get_slide_window()  # Adjust the sliding window size

        self.allStates = self.init_all_states()  # Initialize all states
        self._adjust_close_price()  # Adjust the closing prices based on the sliding window

    def _adjust_close_price(self):
        # Adjusts the closing prices based on the sliding window size
        self.adjustedClosePrice = self.closePrice[self.slideWindow:]

    def _get_slide_window(self):
        # Sets the sliding window size to the larger of the two window sizes
        self.slideWindow = max(self.windowSize)

    def init_all_states(self):
        """
        This function creates an array of all states, where each state is a normalized sliding window of the input data.
        The normalized data ranges between 0 and 1, and can be fed into the machine learning model.
        """
        scaler = MinMaxScaler()
        scaledData = scaler.fit_transform(self.inputData)
        sequences = [scaledData[i:i + self.slideWindow] for i in range(len(scaledData) - self.slideWindow + 1)]
        return np.round(np.array(sequences), decimals=4)

    def percentage_change(self, future):
        """
        Calculates the percentage change in closing prices for a future number of steps.
        The percentage change is calculated relative to the current closing price.
        """
        self.percentageChangeInFuture = self.closePrice.pct_change(future).iloc[future:]
        self.percentageChangeInFuture = self.percentageChangeInFuture.iloc[self.slideWindow:]
        return self.percentageChangeInFuture

    def all_pctC_based_rewards(self):
        """
        Calculates the reward for each state based on the percentage change in closing price.
        If the percentage change is positive (price goes up), the reward for a "buy" action is the percentage change.
        If the percentage change is negative (price goes down), the reward for a "sell" action is the absolute value of
        the percentage change.
        """

        def reward(x):
            return [x * 100, 0] if x > 0 else [0, abs(x * 100)]  # [reward for buy, reward for sell]

        self.rewards = [reward(pctC[0]) for pctC in self.percentageChangeInFuture.values]
        return self.rewards

    def reward_at_time_t(self, t):
        # Returns the reward at a specific time step
        return self.rewards[t]

    def state_at_time_t(self, t):
        # Returns the state at a specific time step
        return self.allStates[t]

    def hold_state(self, holdTime, t):
        """
        Returns a state with an extra dimension that represents how long a stock has been held.
        The "hold time" is represented by a column of ones in the state array.
        If the stock has been held for 5 time steps, for example, there are 5 ones in the column.
        """
        state = self.state_at_time_t(t)
        window = state.shape[0]

        actionColumn = np.ones((window, 1)) if holdTime > window else np.zeros((window, 1))
        if holdTime < window:
            actionColumn[0:holdTime] = 1

        return np.concatenate((state, actionColumn), axis=1)
