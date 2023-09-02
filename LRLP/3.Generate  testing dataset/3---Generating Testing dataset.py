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
from result_util import get_pair_strategy_return



testing_start_index = '2022-09-01'
testing_end_index = '2023-09-1'

BTC = yf.download('BTC-USD', start=testing_start_index, end=testing_end_index) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=testing_start_index, end=testing_end_index)  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)
pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()

pair_feature = pd.concat([BTC[['Open', 'High', 'Low', 'Close','Adj Close', 'Volume']],
                          ETH[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']]], ignore_index=True, axis=1)

# pair_feature.columns=['BTC_Open_R', 'BTC_High_R', 'BTC_Low_R', 'BTC_Close_R', 'BTC_Volume_R',
#                       'ETH_Open_R', 'ETH_High_R', 'ETH_Low_R', 'ETH_Close_R', 'ETH_Volume_R']

# Normalization  Daily Change rate
pair_feature_ratio_shift1=normalize_series(pair_feature.shift(1)).dropna()
pair_feature_ratio_shift1.columns=['BTC_Open_R(-1)', 'BTC_High_R(-1)', 'BTC_Low_R(-1)', 'BTC_Close_R(-1)','BTC_Adj_R(-1)','BTC_Volume_R(-1)',
                                   'ETH_Open_R(-1)', 'ETH_High_R(-1)', 'ETH_Low_R(-1)', 'ETH_Close_R(-1)','ETH_Ajd_R(-1)', 'ETH_Volume_R(-1)']
# Normalization  Daily Change rate
pair_feature_ratio_shift2=normalize_series(pair_feature.shift(2)).dropna()
pair_feature_ratio_shift2.columns=['BTC_Open_R(-2)', 'BTC_High_R(-2)', 'BTC_Low_R(-2)', 'BTC_Close_R(-2)','BTC_Adj_R(-2)', 'BTC_Volume_R(-2)',
                                   'ETH_Open_R(-2)', 'ETH_High_R(-2)', 'ETH_Low_R(-2)', 'ETH_Close_R(-2)','ETH_Ajd_R(-2)', 'ETH_Volume_R(-2)']

# Normalization   Daily Change rate
pair_ret=normalize_series(pair)

#remove first row with NAs
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']


btc_R = pair_ret['BTC_RET']
eth_R = pair_ret['ETH_RET']


# Use hege to get trading pairs
hege= rg.OLS(btc_R, eth_R).fit().params[0]
pair_train= btc_R - hege * eth_R

# returns to build signals
rbtc_ret= pair_ret['BTC_RET']
reth_ret= pair_ret['ETH_RET']

testing_dataset= pd.concat([btc_R ,eth_R,pair_feature_ratio_shift1,pair_feature_ratio_shift2], ignore_index=False,axis=1)
testing_dataset = testing_dataset.dropna()

# -------------------------------The hyperparameter obtained by GA----------------
# a = 1.1730277004674885  #Return  920.9439
# b = 0.8829080292183107
# k = 2
# window1 = 1
# window2 = 79

a = 1.1949552532482448  #1.2585  -0.090287
b = 0.21272475392074522
k = 2
window1 = 1
window2 = 28


# a = 1.6620194524076644  #Return 58.603
# b = 0.7798181445467653
# k = 2
# window1 = 4
# window2 = 77


#---------------------------------0829
# a = 1.4328478353111154   #Return : 471.799
# b = 0.16647979564829982
# k = 2
# window1 = 1
# window2 = 20

# a = 1.6948122466279503 #Return : 40.8333
# b = 0.9265630518121221
# k = 2
# window1 = 4
# window2 = 57

# a = 1.1486201606191768  #Return : 76.6564  Max Drawdown: -0.4982
# b = 0.30927478623887855
# k = 5
# window1 =19
# window2 = 86


# -------------------------------The hyperparameter obtained by GA----------------


z_score,_ = get_z_socre_two_windows(btc_R,eth_R,window1,window2)

z_score = z_score.dropna()


# tests.insert(len(tests.columns), 'ftestsig2', z_score_singel)
testing_dataset.insert(len(testing_dataset.columns), 'rbtc_ret', rbtc_ret)
testing_dataset.insert(len(testing_dataset.columns), 'reth_ret', reth_ret)
testing_dataset.insert(len(testing_dataset.columns), 'z_score', z_score)
testing_dataset.insert(len(testing_dataset.columns), 'z_score(-1)', z_score.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'z_score(-2)', z_score.shift(2))

# No trading signals and profit are generated for test data
testing_dataset.insert(len(testing_dataset.columns), 'z_score_singel_for_lable', -100)
testing_dataset.insert(len(testing_dataset.columns), 'port_outa_z_score_singel_for_lable', -1000)


# 'rbtc_ret(-1)',  'rbtc_ret(-2)',  'reth_ret(-1)','reth_ret(-2)'
# Rate of return change  (return on returns)
rrbtc = (pair_ret['BTC_RET'].pct_change(1).dropna()).pct_change(1).dropna()
rreth = (pair_ret['ETH_RET'].pct_change(1).dropna()).pct_change(1).dropna()



nor_pair_feature=normalize_series(pair_feature).dropna()

nor_pair_feature.columns=['BTC_Open_R', 'BTC_High_R', 'BTC_Low_R', 'BTC_Close_R','BTC_Ajd_R','BTC_Volume_R',
                          'ETH_Open_R', 'ETH_High_R', 'ETH_Low_R', 'ETH_Close_R','ETH_Ajd_R','ETH_Volume_R']
speed_pair_feature=(nor_pair_feature.pct_change(1).dropna()).pct_change(1).dropna()
speed_BTC_Open_R=speed_pair_feature['BTC_Open_R']
speed_BTC_High_R=speed_pair_feature['BTC_High_R']
speed_BTC_Low_R=speed_pair_feature['BTC_Low_R']
speed_BTC_Close_R=speed_pair_feature['BTC_Close_R']
speed_BTC_Volume_R=speed_pair_feature['BTC_Volume_R']
speed_BTC_Adj_R=speed_pair_feature['BTC_Ajd_R']

# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Open_R(-1)', speed_BTC_Open_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Open_R(-2)', speed_BTC_Open_R.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_High_R(-1)', speed_BTC_High_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_High_R(-2)', speed_BTC_High_R.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Low_R(-1)', speed_BTC_Low_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Low_R(-2)', speed_BTC_Low_R.shift(2))
testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Adj_R(-1)', speed_BTC_Adj_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Adj_R(-2)', speed_BTC_Adj_R.shift(2))


testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Close_R(-1)', speed_BTC_Close_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Close_R(-2)', speed_BTC_Close_R.shift(2))
testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Volume_R(-1)', speed_BTC_Volume_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_BTC_Volume_R(-2)', speed_BTC_Volume_R.shift(2))

speed_ETH_Open_R=speed_pair_feature['ETH_Open_R']
speed_ETH_High_R=speed_pair_feature['ETH_High_R']
speed_ETH_Low_R=speed_pair_feature['ETH_Low_R']
speed_ETH_Close_R=speed_pair_feature['ETH_Close_R']
speed_ETH_Volume_R=speed_pair_feature['ETH_Volume_R']
speed_ETH_Adj_R=speed_pair_feature['ETH_Ajd_R']
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Open_R(-1)', speed_ETH_Open_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Open_R(-2)', speed_ETH_Open_R.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_High_R(-1)', speed_ETH_High_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_High_R(-2)', speed_ETH_High_R.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Low_R(-1)', speed_ETH_Low_R.shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Low_R(-2)', speed_ETH_Low_R.shift(2))
testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Adj_R(-1)', speed_ETH_Adj_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Adj_R(-2)', speed_ETH_Adj_R.shift(2))

testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Close_R(-1)', speed_ETH_Close_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Close_R(-2)', speed_ETH_Close_R.shift(2))
testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Volume_R(-1)', speed_ETH_Volume_R.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_ETH_Volume_R(-2)', speed_ETH_Volume_R.shift(2))






# Insert into the trading data set
testing_dataset.insert(len(testing_dataset.columns), 'rbtc_ret(-1)', rbtc_ret.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'rbtc_ret(-2)', rbtc_ret.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'rbtc_ret(-3)', rbtc_ret.shift(3))

testing_dataset.insert(len(testing_dataset.columns), 'reth_ret(-1)', reth_ret.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'reth_ret(-2)', reth_ret.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'reth_ret(-3)', reth_ret.shift(3))

testing_dataset.insert(len(testing_dataset.columns), 'rrbtc(-1)', rrbtc.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'rrbtc(-2)', rrbtc.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'rrbtc(-3)', rrbtc.shift(3))

testing_dataset.insert(len(testing_dataset.columns), 'rreth(-1)', rreth.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'rreth(-2)', rreth.shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'rreth(-3)', rreth.shift(3))

pt_out_pair_trading,pair_trading_dataset=get_pair_strategy_return(testing_start_index,testing_end_index)

# The returns generated by pait trading and logarithmic returns
# testing_dataset.insert(len(testing_dataset.columns), 'Log_R(-1)', pair_trading_dataset['Log_R(-1)'])
# testing_dataset.insert(len(testing_dataset.columns), 'port_out(-1)', pair_trading_dataset['port_out(-1)'])
# testing_dataset.insert(len(testing_dataset.columns), 'Log_R(-2)', pair_trading_dataset['Log_R(-2)'])
# testing_dataset.insert(len(testing_dataset.columns), 'port_out(-2)', pair_trading_dataset['port_out(-2)'])

# testing_dataset.insert(len(testing_dataset.columns), 'up_th(-1)', pair_trading_dataset['up_th'].shift(1))
# testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-1)', pair_trading_dataset['lw_th'].shift(1))

# testing_dataset.insert(len(testing_dataset.columns), 'up_th(-2)', pair_trading_dataset['up_th'].shift(2))
# testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-2)', pair_trading_dataset['lw_th'].shift(2))
#
# testing_dataset.insert(len(testing_dataset.columns), 'up_th(-3)', pair_trading_dataset['up_th'].shift(3))
# testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-3)', pair_trading_dataset['lw_th'].shift(3))
# testing_dataset.insert(len(testing_dataset.columns), 'pair_trading_signal', pair_trading_dataset['ftestsig2'])

slide_window=3
up_th = (z_score.rolling(window=slide_window,min_periods=1).mean()) + (z_score.rolling(window=slide_window,min_periods=1).std() * (a-1))  # upper threshold
lw_th = (z_score.rolling(window=slide_window,min_periods=1).mean()) - (z_score.rolling(window=slide_window,min_periods=1).std() * (1-b))  # lower threshold
up_th = up_th.dropna()
lw_th = lw_th.dropna()
rate_up_th=up_th.pct_change(1).dropna()
rate_lw_th=lw_th.pct_change(1).dropna()
testing_dataset.insert(len(testing_dataset.columns), 'up_th(-1)', rate_up_th.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-1)', rate_lw_th.shift(1))

testing_dataset.insert(len(testing_dataset.columns), 'up_th(-2)', rate_up_th.shift(2))
testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-2)', rate_lw_th.shift(2))
#
testing_dataset.insert(len(testing_dataset.columns), 'up_th(-3)', rate_up_th.shift(3))
testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-3)', rate_up_th.shift(3))

# testing_dataset.insert(len(testing_dataset.columns), 'up_th(-4)', rate_up_th.shift(4))
# testing_dataset.insert(len(testing_dataset.columns), 'lw_th(-4)', rate_up_th.shift(4))

speed_z_score=(z_score.pct_change(1).dropna()).pct_change(1).dropna()
testing_dataset.insert(len(testing_dataset.columns), 'speed_z_score(-1)', speed_z_score.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'speed_z_score(-2)', speed_z_score.shift(2))

rate_z_score=z_score.pct_change(1).dropna()
testing_dataset.insert(len(testing_dataset.columns), 'rate_z_score(-1)', rate_z_score.shift(1))
testing_dataset.insert(len(testing_dataset.columns), 'rate_z_score(-2)', rate_z_score.shift(2))

testing_dataset=testing_dataset.dropna()
testing_dataset.to_csv("../4.Generate y_predict using Adaboost (traing dataset$testing dataset)/0617_testing_dataset.csv", index=True)


print('08311518')