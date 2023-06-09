# dql-trading-bot-training

This project presents an implementation for training a Deep Q-Learning (DQL) agent specifically designed for making informed predictions related to 
optimal buying or selling decisions within the stock market domain. Notably, it provides users with the flexibility to either incorporate their own
custom neural network architecture or leverage the default neural network architecture embedded in the framework.

The "retrain_dql_model" notebook serves the purpose of retraining an already trained model that operates on the AWS cloud infrastructure. 
This notebook is responsible for storing the model parameters in an S3 bucket if the retrained model surpasses the performance of the existing one.
The implementation details of the trading bot, including the integration of DQL, can be accessed through the 
project repository available at: https://github.com/DavidWT93/dql-aws-traidingbot.

For a live demonstration of the model's functionality in executing real-time trades, please visit my webpage at: http://www.davidtillery.xyz/trading_home.
