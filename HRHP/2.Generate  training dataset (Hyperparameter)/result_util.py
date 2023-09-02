import math

import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from mealpy.evolutionary_based import GA
# from mealpy.swarm_based import GWO
# from permetrics.regression import RegressionMetric
# import tensorflow as tf
import statsmodels.regression.linear_model as rg
import numpy as np
import random
random.seed(7)
np.random.seed(42)
# tf.random.set_seed(116)
from GA_util_all_data import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd
from numpy import array, reshape
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from baseline_util import get_z_socre_hege,get_z_socre_no_hege,get_z_socre_two_windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]


def get_pair_strategy_return(traing_start_index,traing_end_index):
    BTC = yf.download('BTC-USD',start=traing_start_index, end=traing_end_index)
    ETH = yf.download('ETH-USD', start=traing_start_index, end=traing_end_index)


    pair = pd.concat([BTC['Adj Close'], ETH['Adj Close']], ignore_index=True, axis=1)
    pair = pair.dropna()


    pair_ret = normalize_series(pair)

    # remove first row with NAs
    pair_ret = pair_ret.tail(len(pair_ret) - 1)
    pair_ret.columns = ['BTC_RET', 'ETH_RET']

    # split into train and validation/testing
    split = 90  # int(len(pair_ret) * 0.7)

    btc_R_train = pair_ret['BTC_RET'][:split]
    btc_R_test = pair_ret['BTC_RET'][split:]
    eth_R_train = pair_ret['ETH_RET'][:split]
    eth_R_test = pair_ret['ETH_RET'][split:]
    tests = pd.concat([btc_R_test, eth_R_test], ignore_index=False, axis=1)

    # calculating for beta (Hedge)
    hege = rg.OLS(btc_R_train, eth_R_train).fit().params[0]  #0.32 #0.32   1.5#
    pair_spread = btc_R_test - hege * eth_R_test


    # Spread Mean and Standard dev
    spread_mean = pair_spread.mean()
    spread_sd = pair_spread.std()
    print('the mean of the spread is', spread_mean)
    print('the Standard Dev of the Spread is', spread_sd)
    # z_score=(pair_spread-spread_mean)/spread_sd
    window = 15  # 29 54
    pair_train = btc_R_test - hege * eth_R_test
    # BTC_ETH Rolling Spread Z-Score Calculation
    z_score = (pair_train - pair_train.rolling(window=window, min_periods=1).mean()) / pair_train.rolling(window=window,
                                                                                                          min_periods=1).std()

    z_score = z_score.dropna()
    # print(z_score.rolling(window=2,min_periods=1).mean())
    z_score_mean = z_score.rolling(window=2, min_periods=1).mean()
    # z_score_mean=z_score_mean[1:]
    # print(z_score.rolling(window=2,min_periods=1).std())
    z_score_std = z_score.rolling(window=2, min_periods=1).std()
    z_score_std.dropna()

    up_th = (z_score.rolling(window=2).mean()) + (z_score.rolling(window=2).std() * 2)  # upper threshold
    lw_th = (z_score.rolling(window=2).mean()) - (z_score.rolling(window=2).std() * 2)  # lower threshold

    up_th = up_th.dropna()
    lw_th = lw_th.dropna()
    up_lw = pd.concat([up_th, lw_th, btc_R_test, eth_R_test, z_score, z_score.rolling(window=2).mean(),
                       z_score.rolling(window=2).std() * 2], ignore_index=True, axis=1)
    up_lw.columns = ['up_th', 'lw_th', 'btc_R_test', 'eth_R_test', 'z_score', 'mean(window=2)', 'deviation(window=2)']
    up_lw.dropna()
    rbtc_ret = pair_ret['BTC_RET'].tail(int(len(pair_ret) - split))  # .tail(int(len(pair_ret) * 0.3))
    reth_ret = pair_ret['ETH_RET'].tail(int(len(pair_ret) - split))   # .tail(int(len(pair_ret) * 0.3))
    # return on returns
    rrbtc = (pair_ret['BTC_RET'].pct_change(1).dropna()).pct_change(1).dropna()

    rreth = (pair_ret['ETH_RET'].pct_change(1).dropna()).pct_change(1).dropna()

    trade_dir = pd.DataFrame(rbtc_ret)
    trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-1)', rbtc_ret.shift(1))
    trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-2)', rbtc_ret.shift(2))
    trade_dir.insert(len(trade_dir.columns), 'reth_ret(-1)', reth_ret.shift(1))
    trade_dir.insert(len(trade_dir.columns), 'reth_ret(-2)', reth_ret.shift(2))
    trade_dir.insert(len(trade_dir.columns), 'rrbtc(-1)', rrbtc.shift(1))
    trade_dir.insert(len(trade_dir.columns), 'rrbtc(-2)', rrbtc.shift(2))
    trade_dir.insert(len(trade_dir.columns), 'rreth(-1)', rreth.shift(1))
    trade_dir.insert(len(trade_dir.columns), 'rreth(-2)', rreth.shift(2))
    trade_dirsig2 = 0.0
    trade_dirsig2a = []

    trade_dir = trade_dir.dropna()

    for i in range(0, len(trade_dir.index)):
        if trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] > (rreth[trade_dir.index[i]]) and trade_dir.at[
            trade_dir.index[i], 'rrbtc(-1)'] < (rreth[trade_dir.index[i]]):
            trade_dirsig2 = 2
        elif trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] < (rreth[trade_dir.index[i]]) and trade_dir.at[
            trade_dir.index[i], 'rrbtc(-1)'] > (rreth[trade_dir.index[i]]):
            trade_dirsig2 = -2
        elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] > (rrbtc[trade_dir.index[i]]) and trade_dir.at[
            trade_dir.index[i], 'rreth(-1)'] < (rrbtc[trade_dir.index[i]]):
            trade_dirsig2 = 1
        elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] < (rrbtc[trade_dir.index[i]]) and trade_dir.at[
            trade_dir.index[i], 'rreth(-1)'] > (rrbtc[trade_dir.index[i]]):
            trade_dirsig2 = -1
        else:
            trade_dirsig2 = 0.0
        trade_dirsig2a.append(trade_dirsig2)

    trade_dir.insert(len(trade_dir.columns), 'trade_dirsig2', trade_dirsig2a)

    # 3.3.5. BTC_ETH Trading Strategy Signals
    tests.insert(len(tests.columns), 'z_score', z_score)
    tests.insert(len(tests.columns), 'z_score(-1)', z_score.shift(1))
    tests.insert(len(tests.columns), 'z_score(-2)', z_score.shift(2))
    tests.insert(len(tests.columns), 'trade_dir', trade_dir['trade_dirsig2'])

    tests = tests.dropna()

    ftestsig2 = 0.0
    ftestsig2a = []
    for i in range(0, len(tests.index)):
        # if tests.at[tests.index[i], 'z_score(-2)'] > (-1 * up_th[tests.index[i]]) and tests.at[
        #     tests.index[i], 'z_score(-1)'] < (-1 * up_th[tests.index[i]]):
        #     ftestsig2 = 1
        # elif tests.at[tests.index[i], 'z_score(-2)'] < (-1 * lw_th[tests.index[i]]) and tests.at[
        #     tests.index[i], 'z_score(-1)'] > (-1 * lw_th[tests.index[i]]):
        #     ftestsig2 = -2
        # elif tests.at[tests.index[i], 'z_score(-2)'] < (-1 * up_th[tests.index[i]]) and tests.at[
        #     tests.index[i], 'z_score(-1)'] > (-1 * up_th[tests.index[i]]):
        #     ftestsig2 = -1
        # elif tests.at[tests.index[i], 'z_score(-2)'] > (-1 * lw_th[tests.index[i]]) and tests.at[
        #     tests.index[i], 'z_score(-1)'] < (-1 * lw_th[tests.index[i]]):
        #     ftestsig2 = 2
        # elif tests.at[tests.index[i], 'z_score(-2)'] < up_th[tests.index[i]] and tests.at[
        #     tests.index[i], 'z_score(-1)'] > up_th[tests.index[i]]:
        #     ftestsig2 = -1
        # elif tests.at[tests.index[i], 'z_score(-2)'] > up_th[tests.index[i]] and tests.at[
        #     tests.index[i], 'z_score(-1)'] < up_th[tests.index[i]]:
        #     ftestsig2 = 1
        # elif tests.at[tests.index[i], 'z_score(-2)'] > lw_th[tests.index[i]] and tests.at[
        #     tests.index[i], 'z_score(-1)'] < lw_th[tests.index[i]]:
        #     ftestsig2 = 2
        # elif tests.at[tests.index[i], 'z_score(-2)'] < lw_th[tests.index[i]] and tests.at[
        #     tests.index[i], 'z_score(-1)'] > lw_th[tests.index[i]]:
        #     ftestsig2 = -2
        if tests.at[tests.index[i], 'trade_dir'] == 1:
            ftestsig2 = 2
        elif tests.at[tests.index[i], 'trade_dir'] == -1:
            ftestsig2 = -2
        elif tests.at[tests.index[i], 'trade_dir'] == 2:
            ftestsig2 = 1
        elif tests.at[tests.index[i], 'trade_dir'] == -2:
            ftestsig2 = -1
        else:
            ftestsig2 = 0.0
        ftestsig2a.append(ftestsig2)

    tests.insert(len(tests.columns), 'rbtc_ret', rbtc_ret)
    tests.insert(len(tests.columns), 'reth_ret', reth_ret)


    port_out = 0.0
    port_outa = []
    tests.insert(len(tests.columns), 'ftestsig2', ftestsig2a)

    for i in range(0, len(tests.index)):
        if tests.at[tests.index[i], 'ftestsig2'] == -2:
            port_out = tests.at[tests.index[i], 'rbtc_ret']
        elif tests.at[tests.index[i], 'ftestsig2'] == -1:
            port_out = tests.at[tests.index[i], 'reth_ret']
        elif tests.at[tests.index[i], 'ftestsig2'] == 2:
            port_out = tests.at[tests.index[i], 'rbtc_ret']
        elif tests.at[tests.index[i], 'ftestsig2'] == 1:
            port_out = tests.at[tests.index[i], 'reth_ret']
        else:
            port_out = 0
        port_outa.append(port_out)
    tests.insert(len(tests.columns), 'port_out', port_outa)


    tests.insert(len(tests.columns), 'Log_R(-1)', np.log(1 + pd.DataFrame(tests['port_out']).shift(1)))
    tests.insert(len(tests.columns), 'port_out(-1)', pd.DataFrame(tests['port_out']).shift(1))
    tests.insert(len(tests.columns), 'Log_R(-2)', np.log(1 + pd.DataFrame(tests['port_out']).shift(2)))
    tests.insert(len(tests.columns), 'port_out(-2)', pd.DataFrame(tests['port_out']).shift(2))

    tests.insert(len(tests.columns), 'up_th', up_th)
    tests.insert(len(tests.columns), 'lw_th', lw_th)

    tests = tests.fillna(method='ffill')

    pt_out_pair_trading = (1 + tests['port_out']).cumprod()
    return pt_out_pair_trading,tests

#
# pt_out_pair_trading,_=get_pair_strategy_return()
#
# print('the MDD is', get_mdd(pt_out_pair_trading))
#
# plt.figure(figsize=(16,8))
# plt.rcParams.update({'font.size':10})
# plt.xticks(rotation=45)
# ax = plt.axes()
# ax.xaxis.set_major_locator(plt.MaxNLocator(20))
# # plt.plot(pt_out_pair_trading, label='Cumulative return on P-Trading Strategy portfolio',color='b')
# plt.plot(pt_out_pair_trading, label='Cumulative return on P-Trading Strategy',color='b')
# # plt.plot(port_z_scorec, label='Cumulative return on Labeling_+5%Cm',color='y')
# # plt.plot(pt_outc, label='Cumulative return on P-Trading Strategy_+5%Cm')
# # plt.plot(bh_btc, label='Cumulative return on Buy and Hold Bitcoin',color='g')
# # plt.plot(bh_eth, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
# plt.title('Labeling Method Cumulative Returns')
# plt.xlabel("Date")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.suptitle('Labeling Method  Cumulative Returns  (120 days ( 55-days rolling window))')
# ax.legend(loc='best')
# ax.grid(True)
# plt.show()
#
#
