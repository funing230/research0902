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

def get_z_socre_hege(btc_R_train, eth_R_train,btc_R_test,eth_R_test,window1,window2) :
    hege = rg.OLS(btc_R_train, eth_R_train).fit().params[0]
    pair_train = btc_R_test - hege * eth_R_test
    z_score= (pair_train - pair_train.rolling(window=window1, min_periods=1).mean()) / \
             pair_train.rolling(window=window1,min_periods=1).std()

    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)
    function_name = frame_info.function
    return  z_score,function_name

def get_z_socre_no_hege(btc_R_train, eth_R_train,btc_R_test,eth_R_test,window1,window2) :
    pair_train = btc_R_test - eth_R_test
    z_score= (pair_train - pair_train.rolling(window=window1, min_periods=1).mean()) / \
             pair_train.rolling(window=window1,min_periods=1).std()
    frame = inspect.currentframe()
    frame_info = inspect.getframeinfo(frame)
    function_name = frame_info.function
    return  z_score,function_name

def get_z_socre_two_windows(btc_R_test,eth_R_test,window1,window2) :
    pair_train = btc_R_test / eth_R_test
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
    计算给定日收益率序列的夏普比率。

    :param daily_returns: NumPy数组或列表，代表日收益率。
    :param risk_free_rate: 年化无风险利率，默认值为1%。
    :return: 计算出的夏普比率。
    """
    # 将日收益率列表转换为NumPy数组（如果尚未转换）
    daily_returns = np.array(daily_returns)

    # 计算日收益率的平均值和标准差
    average_daily_return = np.mean(daily_returns)
    std_dev_daily_return = np.std(daily_returns)

    # 年化收益率和标准差
    average_annual_return = average_daily_return * 250  # 假设一年有250个交易日
    std_dev_annual_return = std_dev_daily_return * np.sqrt(250)

    # 计算夏普比率
    sharpe_ratio = (average_annual_return - risk_free_rate) / std_dev_annual_return

    return sharpe_ratio
# totaldataset_file_path = 'totaldataset_df_BTC.csv'
# BTC_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
# totaldataset_file_path = 'totaldataset_df_ETH.csv'
# ETH_df = pd.read_csv(totaldataset_file_path, parse_dates=[0], index_col=0)
#
#
# # pair= pd.concat([BTC_df['close'],ETH_df['close']], ignore_index=True,axis=1)
# #
# # BTC_df_1 = pd.DataFrame({'close': BTC_df['close']})
# # ETH_df_1 = pd.DataFrame({'close': ETH_df['close']})
# # BTC_df_vol= getDailyVol(BTC_df_1)
# # ETH_df_vol= getDailyVol(ETH_df_1)
#
# BTC_df_ret = triple_barrier(BTC_df['close'], 1.07, 0.97, 20)
# ETH_df_ret = triple_barrier(ETH_df['close'], 1.07, 0.97, 20)
#
#
#
#
#
#
# result = pd.concat([BTC_df_ret['triple_barrier_signal'], ETH_df_ret['triple_barrier_signal']], axis=1)