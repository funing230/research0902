# from Pairs_Trading_Deep_Reinforcement_Learning_GPT import PairsTradingDQN
import yfinance as yf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
random.seed(7)
np.random.seed(42)

class PairsTradingDQN:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 3  # 0: hold, 1: long stock1/short stock2, 2: short stock1/long stock2
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio for a given series of daily returns.

    :param daily_returns: A NumPy array or list representing daily returns.
    :param risk_free_rate: The annualized risk-free rate, default value is 1%.
    :return: The calculated Sharpe Ratio.
    """
    # Convert the list of daily returns to a NumPy array (if not already converted)
    daily_returns = np.array(daily_returns)

    # Calculate the mean and standard deviation of daily returns
    average_daily_return = np.mean(daily_returns)
    std_dev_daily_return = np.std(daily_returns)

    # Annualize the returns and standard deviation
    average_annual_return = average_daily_return * 365  # Assuming there are 365 trading days in a year
    std_dev_annual_return = std_dev_daily_return * np.sqrt(365)

    # Calculate the Sharpe Ratio
    sharpe_ratio = (average_annual_return - risk_free_rate) / std_dev_annual_return

    return sharpe_ratio

def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (peak_upper, peak_lower, mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]


# 1. Descargar datos
start_date = "2017-11-09"
end_date = "2022-08-31"
pairs = [('BTC-USD', 'ETH-USD')]

N = 5  # Window size for state
# 加载模型
agent = PairsTradingDQN(2*N)
agent.model.load_weights ("model_weights.h5")         #  ("model_weights.h5")       #("path_to_save_your_weights.h5")




data = {}
for stock1, stock2 in pairs:
    data[stock1] = yf.download(stock1, start=start_date, end=end_date)
    data[stock2] = yf.download(stock2, start=start_date, end=end_date)


# def backtest_strategy(stock1_data, stock2_data, signals):
#     # Calculate daily returns
#     stock1_data['Daily Return'] = stock1_data['Close'].pct_change()
#     stock2_data['Daily Return'] = -stock2_data['Close'].pct_change()
#
#     stock1_data['Strategy Return'] = np.nan
#     stock1_data['Strategy Return'][N:] = stock1_data['Daily Return'][N:] * np.array(signals)
#
#     stock2_data['Strategy Return'] = np.nan
#     stock2_data['Strategy Return'][N:] = stock2_data['Daily Return'][N:] * np.array(signals)
#
#     stock1_data['Cumulative Return'] = (1 + stock1_data['Strategy Return']).cumprod()
#     stock2_data['Cumulative Return'] = (1 + stock2_data['Strategy Return']).cumprod()
#     portfolio_cumulative_return = stock1_data['Cumulative Return'] + stock2_data['Cumulative Return']
#     # Visualize cumulative returns
#     plt.figure(figsize=(10,5))
#     plt.plot(stock1_data.index[N:], portfolio_cumulative_return[N:], label="Pairs Trading Strategy")
#     plt.legend()
#     plt.title(f"Pairs Trading Strategy Cumulative Returns for {stock1}/{stock2}")
#     plt.show()
#

def backtest_strategy(stock1_data, stock2_data, signals, N):
    # Calculate daily returns
    stock1_data['Daily Return'] = stock1_data['Close'].pct_change()
    stock2_data['Daily Return'] = stock2_data['Close'].pct_change()

    # Initialize an array to store daily strategy return
    strategy_returns = []

    # Ensure the length of signals is the same as the length of stock data after dropping NaN
    signals = signals[:len(stock1_data) - N]

    for i in range(len(signals)):
        if signals[i] == 1:  # Buy stock1
            daily_return = stock1_data['Daily Return'].iat[i + N]
        elif signals[i] == 2:  # Buy stock2
            daily_return = stock2_data['Daily Return'].iat[i + N]
        else:  # Hold (cash)
            daily_return = 0
        strategy_returns.append(daily_return)

    # Convert the list to a pandas Series for easy cumulative calculation
    strategy_returns = pd.Series(strategy_returns, index=stock1_data.index[N:])

    # Calculate cumulative return
    cumulative_return = (1 + strategy_returns).cumprod()

    cumulative_return.to_csv('cumulative_return_reinforcement.csv', index=True)

    print("Final Cumulative Return:", cumulative_return[-1])
    print("Sharpe Ratio:", calculate_sharpe_ratio(strategy_returns, 0))
    print("Maximum Drawdown:", get_mdd(cumulative_return))


    # Visualize cumulative returns
    plt.figure(figsize=(10,5))
    plt.plot(cumulative_return, label="Strategy Cumulative Return")
    plt.legend()
    plt.title("Backtest Cumulative Returns")
    plt.show()

    return cumulative_return


# 生成交易信号
signals_after_retraining = {}
for stock1, stock2 in pairs:
    stock1_data = data[stock1]['Close'].values
    stock2_data = data[stock2]['Close'].values
    state = np.concatenate([stock1_data[:N], stock2_data[:N]])
    state = np.reshape(state, [1, 2*N])
    signal = []
    for t in range(N, len(stock1_data)):
        action = agent.act(state)
        signal.append(action)
        next_state = np.concatenate([stock1_data[t-N+1:t+1], stock2_data[t-N+1:t+1]])
        next_state = np.reshape(next_state, [1, 2*N])
        state = next_state
    signals_after_retraining[(stock1, stock2)] = signal

# 重新运行回测
for stock1, stock2 in pairs:
    stock1_data = data[stock1].copy()
    stock2_data = data[stock2].copy()
    backtest_strategy(stock1_data, stock2_data, signals_after_retraining[(stock1, stock2)],N)



#
# def calculate_returns_and_mdd(stock1_prices, stock2_prices, signals):
#     # 计算日收益率
#     stock1_returns = stock1_prices['Close'].pct_change()
#     stock2_returns = stock2_prices['Close'].pct_change()
#
#     # stock1_data['Daily Return'] = stock1_prices['Close'].pct_change()
#     # stock2_data['Daily Return'] = -stock2_prices['Close'].pct_change()
#
#     # 初始化投资组合价值和最大回撤
#     portfolio_value = 1.0
#     max_portfolio_value = portfolio_value
#     mdd = 0.0
#     cumulative_returns = []
#
#     # 确保signals的长度与日收益率数组长度相同
#     signals = signals[1:]
#
#     for i in range(len(signals)):
#         action = signals[i]
#         if action == 1:  # 买入股票1/卖出股票2
#             daily_return = stock1_returns[i] - stock2_returns[i]
#         elif action == 2:  # 卖出股票1/买入股票2
#             daily_return = stock2_returns[i] - stock1_returns[i]
#         else:  # 持有
#             daily_return = 0
#
#         portfolio_value *= (1 + daily_return)
#         cumulative_returns.append(portfolio_value)
#
#         # 更新最大回撤
#         max_portfolio_value = max(max_portfolio_value, portfolio_value)
#         drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
#         mdd = max(mdd, drawdown)
#
#     # 输出最终的累计收益和最大回撤
#     final_cumulative_return = cumulative_returns[-1] if cumulative_returns else None
#     return final_cumulative_return, mdd
#
#
# for stock1, stock2 in pairs:
#     stock1_data = data[stock1].copy()
#     stock2_data = data[stock2].copy()
#     final_return, max_drawdown = calculate_returns_and_mdd(stock1_data, stock2_data, signals_after_retraining[(stock1, stock2)])
#     print("Final Cumulative Return:", final_return)
#     print("Maximum Drawdown:", max_drawdown)
#
#
