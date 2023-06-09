from tqdm import tqdm
import json


def dQL_model_training(window_size, episodes, batch_size, data_samples, dqlModel, data, closePrice,
                       TradingEnvironment, saveEachXEpisode=5, saveModelParamsAs="model", stepsBeforeTraining=60):
    """
    The function DQLModelTraining is designed to train a deep Q-learning model for a trading bot. The model is trained
    using a custom environment, created using the TradingEnvironment class. The training is performed over a number of
     episodes, with each episode corresponding to a complete pass over the historical data.

    For each state, an action is selected by the trading bot and a reward is calculated. This reward, together with the
    state, the action, and the subsequent state, are stored in the trading bot's memory.

    Every 49 steps, the trading bot updates its model using a batch of data from its memory. This is the essential step
    of Q-learning, where the bot learns to associate states and actions with rewards.

    The function also handles logging of the trading bot's performance, storing detailed stats for each episode in a
    JSON file, and periodically saving the weights of the model and plotting the bot's buy and sell decisions. The final
    output is a subplot showing the total profit and total reward per episode.

    Parameters
    ----------
    window_size
    episodes
    batch_size
    data_samples
    dqlModel
    data
    closePrice
    TradingEnvironment

    Returns
    -------

    """
    trainedModelName = None
    # Create a trading environment with the given data and window size
    env = TradingEnvironment(data, window_size, closePrice)

    # Initialize the environment's percentage change and reward methods
    env.percentage_change(3)
    env.all_pctC_based_rewards()

    # Extract all states and rewards from the environment
    allStates = env.allStates
    allRewards = env.all_pctC_based_rewards()
    adjustedClosePrice = env.adjustedClosePrice

    # Start iterating over all episodes
    for episode in range(1, episodes + 1):
        pricePlot, trades, totalProfit, rewardPlot = [], [], [], []
        print("Episode: {}/{}".format(episode, episodes))

        # Initialize the stats for the trader
        traderStats = {
            "trades": [],
            "rewards": [],

            "timesWhenBought": [0],
            "timesWhenSouled": [0],
            "totalProfit": 0,
            "totalPtctC": 0,
        }

        tradeSteps = 0
        bought = False

        # Iterate over all states except the last one
        for t in tqdm(range(len(allStates) - 1)):
            # Fetch the current state and the next state
            state = env.state_at_time_t(t)
            if t == len(allRewards) - 1:
                break
            else:
                next_state = env.state_at_time_t(t + 1)

            # Fetch the action for the current state from the trader
            action = dqlModel.trade(state)

            # Calculate the reward based on the action and update the stats for the trader
            reward, bought, tradeSteps = reward_trading_action(action, t, adjustedClosePrice, traderStats,
                                                               allRewards, bought, tradeSteps)
            traderStats["rewards"].append(reward)

            # Check if we've reached the end of the episode
            done = t == data_samples - 1

            # Append the current state, action, reward, next state, and done flag to the trader's memory
            dqlModel.memory.append((state, action, reward, next_state, done))

            # Every 49 steps, train the trader with a batch from its memory
            if t % stepsBeforeTraining == 0 and len(dqlModel.memory) > batch_size:
                print("EPISODE: {}".format(episode))
                print("batch train")
                print("tradeSteps", tradeSteps)
                dqlModel.batch_train(batch_size)

        if episode % saveEachXEpisode == 0:
            # After x episode,
            # save the trader's stats to a JSON file
            with open(f'trader_stats_episode_{episode}_model_v1.json', "w") as file:
                json.dump(traderStats, file)

            # save the trader's model params
            trainedModelName = f"{saveModelParamsAs}_episode_{episode}.h5"
            dqlModel.model.save(trainedModelName)

    return trainedModelName


def reward_trading_action(action, t, closePrice, traderStats, allRewards, bought, tradeSteps):
    """Calculates the reward for a trading action and updates trading stats.

    Parameters
    ----------
    action : int
        The trading action taken; 0 indicates buying, 1 indicates selling.
    t : int
        The time step at which the action is taken.
    closePrice : pd.Series
        A series object representing the closing prices of the stock.
    traderStats : dict
        A dictionary to store statistics about the trades performed.
    allRewards : list
        A list of the rewards.
    bought : bool
        A flag indicating if a stock has been bought.
    tradeSteps : int
        The number of steps since the last buying action.

    Returns
    -------
    reward : float
        The reward for the current action.
    bought : bool
        Updated bought flag after current action.
    tradeSteps : int
        Updated tradeSteps after current action.
    """

    # Handling the 'buy' action
    if action == 0:
        if not tradeSteps:
            traderStats["timesWhenBought"].append(t)
        tradeSteps += 1
        bought = True
        print(f"AI Trader bought at: {closePrice.iloc[t]}")
        reward = allRewards[t][0]

    # Handling the 'sell' action
    elif action == 1:
        tradeSteps = 0
        reward = allRewards[t][1]

        if bought:
            traderStats["timesWhenSouled"].append(t)
            traderStats["timesWhenSouled"].append(t)
            buy_price = closePrice.iloc[traderStats["timesWhenBought"][-1]]
            sell_price = closePrice.iloc[t]
            currentTradeProfit = round(sell_price - buy_price, 2)
            currentTradeProfitInPctC = round(currentTradeProfit / buy_price * 100, 2)
            traderStats["totalProfit"] += currentTradeProfit.values[0]
            print("AI Trader sold at: ", sell_price,
                  " Profit: " + str(float(currentTradeProfit)))
            traderStats["totalPtctC"] += currentTradeProfitInPctC.values[0]

            traderStats["trades"].append(
                {"trade_profit": currentTradeProfit.values[0], "buy_price": buy_price.values[0],
                 "sell_price": sell_price.values[0],
                 "time_when_bought": traderStats["timesWhenBought"][-1],
                 "time_when_sold": t,
                 "total_profit": traderStats["totalProfit"],
                 "PctC_return": currentTradeProfitInPctC.values[0]})

            print("#################################################")
            print("total profit in episode", round(traderStats["totalProfit"], 2))
            print("#################################################")
            bought = False

    # Handling incorrect action input
    else:
        print("Incorrect action input, expected 0 (buy) or 1 (sell).")
        return False, 0, 0

    return reward, bought, tradeSteps
