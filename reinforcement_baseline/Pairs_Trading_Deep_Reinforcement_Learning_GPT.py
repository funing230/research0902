import yfinance as yf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Descargar datos
start_date = "2017-11-09"
end_date = "2023-12-31"
pairs = [('BTC-USD', 'ETH-USD')]

data = {}
for stock1, stock2 in pairs:
    data[stock1] = yf.download(stock1, start=start_date, end=end_date)
    data[stock2] = yf.download(stock2, start=start_date, end=end_date)

# 2. Definir el agente DQN
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

# 3. Entrenar el agente DQN
N = 5  # Window size for state
agent = PairsTradingDQN(2*N)
episodes = 200

for episode_count, (stock1, stock2) in enumerate(pairs):
    stock1_data = data[stock1]['Close'].values
    stock2_data = data[stock2]['Close'].values

    for episode in range(episodes):
        state = np.concatenate([stock1_data[:N], stock2_data[:N]])
        state = np.reshape(state, [1, 2*N])
        total_reward = 0

        for t in range(N, len(stock1_data) - 1):
            action = agent.act(state)
            next_state = np.concatenate([stock1_data[t-N+1:t+1], stock2_data[t-N+1:t+1]])
            next_state = np.reshape(next_state, [1, 2*N])

            if action == 1:  # long stock1/short stock2
                reward = stock1_data[t+1] / stock1_data[t] - stock2_data[t+1] / stock2_data[t]
            elif action == 2:  # short stock1/long stock2
                reward = stock2_data[t+1] / stock2_data[t] - stock1_data[t+1] / stock1_data[t]
            else:
                reward = 0

            total_reward += reward
            done = t == len(stock1_data) - 2

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent.replay()

        # Print information
        print(f"Episode {episode+1}/{episodes} for stock pair {stock1}/{stock2} completed. Total Reward: {total_reward:.4f}")

    # Print final training information for each stock pair
    print(f"Final training for pair {stock1}/{stock2} completed after {episodes} episodes.")

# 4. Generate trading signals
signals = {}
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
    signals[(stock1, stock2)] = signal

agent.model.save_weights("model_weights1220.h5")

# 5. Backtest
def backtest_strategy(stock1_data, stock2_data, signals):
    # Calculate daily returns
    stock1_data['Daily Return'] = stock1_data['Close'].pct_change()
    stock2_data['Daily Return'] = -stock2_data['Close'].pct_change()

    stock1_data['Strategy Return'] = np.nan
    stock1_data['Strategy Return'][N:] = stock1_data['Daily Return'][N:] * np.array(signals)

    stock2_data['Strategy Return'] = np.nan
    stock2_data['Strategy Return'][N:] = stock2_data['Daily Return'][N:] * np.array(signals)

    stock1_data['Cumulative Return'] = (1 + stock1_data['Strategy Return']).cumprod()
    stock2_data['Cumulative Return'] = (1 + stock2_data['Strategy Return']).cumprod()
    portfolio_cumulative_return = stock1_data['Cumulative Return'] + stock2_data['Cumulative Return']

    # Visualize cumulative returns
    plt.figure(figsize=(10,5))
    plt.plot(stock1_data.index[N:], portfolio_cumulative_return[N:], label="Pairs Trading Strategy")
    plt.legend()
    plt.title(f"Pairs Trading Strategy Cumulative Returns for {stock1}/{stock2}")
    plt.show()

# Run backtest
for stock1, stock2 in pairs:
    stock1_data = data[stock1].copy()
    stock2_data = data[stock2].copy()
    backtest_strategy(stock1_data, stock2_data, signals[(stock1, stock2)])


agent.model.save_weights("path_to_save_your_weights1220.h5")

# Asume que ya has cargado y preprocesado todos tus datos

# Carga el modelo previamente entrenado
agent.model.load_weights("path_to_save_your_weights1220.h5")

# Número de episodios para reentrenamiento
retraining_episodes = 100

for pair_index, (stock1, stock2) in enumerate(pairs):
    stock1_data = data[stock1]['Close'].values
    stock2_data = data[stock2]['Close'].values

    for episode in range(retraining_episodes):
        state = np.concatenate([stock1_data[:N], stock2_data[:N]])
        state = np.reshape(state, [1, 2*N])
        total_reward = 0

        for t in range(N, len(stock1_data) - 1):
            action = agent.act(state)
            next_state = np.concatenate([stock1_data[t-N+1:t+1], stock2_data[t-N+1:t+1]])
            next_state = np.reshape(next_state, [1, 2*N])

            if action == 1:  # long stock1/short stock2
                reward = stock1_data[t+1] / stock1_data[t] - stock2_data[t+1] / stock2_data[t]
            elif action == 2:  # short stock1/long stock2
                reward = stock2_data[t+1] / stock2_data[t] - stock1_data[t+1] / stock1_data[t]
            else:
                reward = 0

            total_reward += reward
            done = t == len(stock1_data) - 2

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent.replay()

        # 打印每个周期的总奖励
        print(f"Pair {stock1}/{stock2}, Episode {episode+1}/{retraining_episodes} completed. Total Reward: {total_reward:.4f}")

    # 打印股票对的训练完成信息
    print(f"Retraining for pair {stock1}/{stock2} completed after {retraining_episodes} episodes.")


# Guarda los pesos del modelo después del reentrenamiento
agent.model.save_weights("path_to_save_your_weights1220.h5")


# # Experimental Setup del PAPER

# In[ ]:




