import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

class PairsTradingDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
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

    def replay(self, batch_size):
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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# This is just a basic outline. For a complete implementation, we would need to add functions to handle the pairs trading logic,
# data preprocessing, training loop, etc.

start_date = "2020-01-01"  # Define tu fecha de inicio aquí
end_date = "2023-12-31"    # Define tu fecha de fin aquí

# Lista de pares
pairs = [('BTC-USD', 'ETH-USD')]

data = {}

for stock1, stock2 in pairs:
    data[stock1] = yf.download(stock1, start=start_date, end=end_date)
    data[stock2] = yf.download(stock2, start=start_date, end=end_date)


def get_signal(stock1_data, stock2_data):
    # Calcular el spread entre los dos activos usando una regresión
    X = stock1_data['Close'].values
    Y = stock2_data['Close'].values
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    stock1_data['Spread'] = Y - model.predict(X)

    # Calcular la media y la desviación estándar del spread
    mean_spread = stock1_data['Spread'].mean()
    std_spread = stock1_data['Spread'].std()

    # Definir umbrales para señales de compra/venta
    z_upper = 1.0  # este valor puede ser ajustado
    z_lower = -1.0 # este valor puede ser ajustado

    upper_threshold = mean_spread + z_upper * std_spread
    lower_threshold = mean_spread + z_lower * std_spread

    # Crear señales de trading basadas en el spread
    stock1_data['Signal'] = 0
    stock1_data['Signal'][stock1_data['Spread'] > upper_threshold] = -1
    stock1_data['Signal'][stock1_data['Spread'] < lower_threshold] = 1

    return stock1_data['Signal']


def backtest_strategy(stock1_data, stock2_data, signals):
    # Calcular los rendimientos diarios
    stock1_data['Daily Return'] = stock1_data['Close'].pct_change() * signals.shift(1)
    stock2_data['Daily Return'] = -stock2_data['Close'].pct_change() * signals.shift(1)

    # Calcular el rendimiento acumulado
    stock1_data['Cumulative Return'] = (1 + stock1_data['Daily Return']).cumprod()
    stock2_data['Cumulative Return'] = (1 + stock2_data['Daily Return']).cumprod()
    portfolio_cumulative_return = stock1_data['Cumulative Return'] + stock2_data['Cumulative Return']

    # Visualizar el rendimiento acumulado
    plt.figure(figsize=(10,5))
    plt.plot(stock1_data.index, portfolio_cumulative_return, label="Pairs Trading Strategy")
    plt.legend()
    plt.title("Pairs Trading Strategy Cumulative Returns")
    plt.show()

# Ejemplo de uso
for stock1, stock2 in pairs:
    stock1_data = data[stock1]
    stock2_data = data[stock2]
    signals = get_signal(stock1_data, stock2_data)
    backtest_strategy(stock1_data, stock2_data, signals)


# # EL MODELO