import numbers
import math
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as rg
import inspect

def triple_barrier(price, ub, lb, max_period):
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0] / s[0]

    r = np.array(range(max_period))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period - 1)[0]

    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period + 1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period + 1)
    t = pd.Series([t.index[int(k + i)] if not math.isnan(k + i) else np.datetime64('NaT')
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1
    signal.loc[p < lb] = -1
    ret = pd.DataFrame({'triple_barrier_profit': p, 'triple_barrier_sell_time': t, 'triple_barrier_signal': signal})

    return ret

def get_z_socre_hege(btc_R, eth_R,window1,window2) :
    hege = rg.OLS(btc_R, eth_R).fit().params[0]
    pair_train = btc_R - hege * eth_R
    z_score= (pair_train - pair_train.rolling(window=window1, min_periods=1).mean()) / \
             pair_train.rolling(window=window1,min_periods=1).std()

    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)
    function_name = frame_info.function
    return  z_score,function_name

def get_z_socre_no_hege(btc_R, eth_R,window1,window2) :
    pair_train = btc_R - eth_R
    z_score= (pair_train - pair_train.rolling(window=window1, min_periods=1).mean()) / \
             pair_train.rolling(window=window1,min_periods=1).std()
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)
    function_name = frame_info.function
    return  z_score,function_name

def get_z_socre_two_windows(btc_R, eth_R,window1,window2) :
    pair_train = btc_R / eth_R
    ma1 = pair_train.rolling(window=window1, center=False,min_periods=1).mean()
    ma2 = pair_train.rolling(window=window2, center=False,min_periods=1).mean()
    std = pair_train.rolling(window=window2, center=False,min_periods=1).std()
    z_score = (ma1 - ma2) / std
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)
    function_name = frame_info.function
    return  z_score,function_name

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
