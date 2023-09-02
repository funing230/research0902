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

