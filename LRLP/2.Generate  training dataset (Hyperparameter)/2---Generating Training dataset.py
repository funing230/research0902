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
from GA_util_all_data import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd,triple_barrier_change_rate
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


# 'Open', 'High', 'Low', 'Close', 'Volume','Adj Close'

# Training period
traing_start_index = '2017-11-09'
traing_end_index = '2022-09-01'

# traing_start_index,traing_end_index

BTC = yf.download('BTC-USD', start=traing_start_index, end=traing_end_index) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=traing_start_index, end=traing_end_index)  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)

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

training_dataset= pd.concat([btc_R ,eth_R,pair_feature_ratio_shift1,pair_feature_ratio_shift2], ignore_index=False,axis=1)
training_dataset = training_dataset.dropna()

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

#
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

# a = 1.6948122466279503 #Return : 40.8333   Max Drawdown: -0.3835
# b = 0.9265630518121221
# k = 2
# window1 = 4
# window2 = 57
#~~~~~~~~~~~~~~~~~~~~~~~~
# a = 1.1486201606191768  #Return : 76.6564  Max Drawdown: -0.4982
# b = 0.30927478623887855
# k = 5
# window1 =19
# window2 = 86



# -------------------------------The hyperparameter obtained by GA----------------

# btc_R ,eth_R

z_score,_ = get_z_socre_two_windows(btc_R,eth_R,window1,window2)

z_score = z_score.dropna()

# z_score_ret=triple_barrier(z_score,a, b, k)

z_score_ret=triple_barrier_change_rate(z_score,rbtc_ret,reth_ret ,a, b, k)


z_score_singel_for_lable = z_score_ret['triple_barrier_signal']
z_score_singel_for_lable.dropna()
# tests.insert(len(tests.columns), 'ftestsig2', z_score_singel)
training_dataset.insert(len(training_dataset.columns), 'rbtc_ret', rbtc_ret)
training_dataset.insert(len(training_dataset.columns), 'reth_ret', reth_ret)
training_dataset.insert(len(training_dataset.columns), 'z_score', z_score)
training_dataset.insert(len(training_dataset.columns), 'z_score(-1)', z_score.shift(1))
training_dataset.insert(len(training_dataset.columns), 'z_score(-2)', z_score.shift(2))
training_dataset.insert(len(training_dataset.columns), 'z_score_singel_for_lable', z_score_singel_for_lable)

# 3.3.6 Trading Strategy Signals, without commission/exchange fee

port_out_z_score_singel_for_lable = 0.0
port_outa_z_score_singel_for_lable = []

for i in range(0, len(training_dataset.index)):
    if training_dataset.at[training_dataset.index[i], 'z_score_singel_for_lable'] == 1:
        '''
        If the value of the z-score touches the upper threshold, 
        indicating a positive deviation from the mean, 
        it means that the growth rate of BTC is too fast. Therefore,
        it is recommended to buy ETH.
        '''
        port_out_z_score_singel_for_lable = training_dataset.at[training_dataset.index[i], 'reth_ret']
    elif training_dataset.at[training_dataset.index[i], 'z_score_singel_for_lable'] == -1:
        '''
        If the value of z_score touches the lower barrier, 
        indicating a negative deviation from the mean, 
        it means that the growth rate of ETH is too fast. 
        Therefore, buy BTC.
        '''
        port_out_z_score_singel_for_lable = training_dataset.at[training_dataset.index[i], 'rbtc_ret']
    else:
        port_out_z_score_singel_for_lable = 0
    port_outa_z_score_singel_for_lable.append(port_out_z_score_singel_for_lable)

# tests_for_lable.insert(len(tests_for_lable.columns), 'Log_R', np.log(1 + pd.DataFrame(port_outa_z_score_singel_for_lable)))
training_dataset.insert(len(training_dataset.columns), 'port_outa_z_score_singel_for_lable', port_outa_z_score_singel_for_lable)
training_dataset = training_dataset.fillna(method='ffill')

port_outa_z_score_singel_for_lable = (1 + training_dataset['port_outa_z_score_singel_for_lable']).cumprod() # pair trading return


# pt_out = port_outa_z_score_singel.iloc[3:]
MDD = get_mdd(port_outa_z_score_singel_for_lable)
print("------FINALLY---------------------------------------")
print("Return : " + str(np.round(port_outa_z_score_singel_for_lable.iloc[-1], 4)))
print("Standard Deviation : " + str(
    np.round(np.std(port_outa_z_score_singel_for_lable), 4)))  # mean_absolute_percentage_error
print("Sharpe Ratio (Rf=0%) : " + str(
    np.round(port_outa_z_score_singel_for_lable.iloc[-1] / (np.std(port_outa_z_score_singel_for_lable)), 4)))
print("Max Drawdown: " + str(np.round(MDD, 4)))  # calculate_mdd(pt_out)
print('++++++++++++++++++++++++++++++++++++++')
print("a : " + str(a))
print("b : " + str(b))
print("k : " + str(k))
print("window1 : " + str(window1))
print("window2 : " + str(window2))

pt_out_pair_trading,pair_trading_dataset=get_pair_strategy_return(traing_start_index,traing_end_index)

bh_btc= (1 + training_dataset['rbtc_ret']).cumprod()
bh_eth= (1 + training_dataset['reth_ret']).cumprod()

plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(pt_out_pair_trading, label='Cumulative return on P-Trading Strategy portfolio',color='b')
plt.plot(port_outa_z_score_singel_for_lable, label='Labeling Method',color='r')
# plt.plot(port_z_scorec, label='Cumulative return on Labeling_+5%Cm',color='y')
# plt.plot(pt_outc, label='Cumulative return on P-Trading Strategy_+5%Cm')
plt.plot(bh_btc, label='Cumulative return on Buy and Hold Bitcoin',color='g')
plt.plot(bh_eth, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
plt.title('Labeling Method VS. P-Trading Strategy Cumulative Returns')
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.suptitle('Labeling Method VS. P-Trading Strategy Cumulative Returns')
ax.legend(loc='best')
ax.grid(True)
plt.show()

results2 = [{'0': 'Test:',
             '1': 'P-Trading Strategy',
             '2': 'triple barrier labeling',
             '3': 'Buy&Hold Bitcoin',
             '4': 'Buy&Hold Ethereum',},

            {'0': 'Return',
             '1': np.round(pt_out_pair_trading.iloc[-1], 4),
             '2': np.round(port_outa_z_score_singel_for_lable.iloc[-1], 4),
             '3': np.round(bh_btc.iloc[-1], 4),
             '4': np.round(bh_eth.iloc[-1], 4),
             # '6': np.round(pt_outc.iloc[-1], 4)
             },

            {'0': 'Standard Deviation',
             '1': np.round(np.std(pt_out_pair_trading), 4),
             '2': np.round(np.std(port_outa_z_score_singel_for_lable), 4),
             '3': np.round(np.std(bh_btc), 4),
             '4': np.round(np.std(bh_eth), 4),
             # '6': np.round(np.std(pt_outc), 4)
             },

            {'0': 'Sharpe Ratio (Rf=0%)',
             '1': np.round(pt_out_pair_trading.iloc[-1] / (np.std(pt_out_pair_trading)), 4),
             '2': np.round(port_outa_z_score_singel_for_lable.iloc[-1] / (np.std(port_outa_z_score_singel_for_lable)), 4),
             '3': np.round(bh_btc.iloc[-1] / (np.std(bh_btc)), 4),
             '4': np.round(bh_eth.iloc[-1] / (np.std(bh_eth)), 4),
             # '6': np.round(pt_outc.iloc[-1] / (np.std(pt_outc)), 4)
             },

            {'0': 'Max Drawdown',
             '1': np.round(get_mdd(pt_out_pair_trading), 4),
             '2': np.round(get_mdd(port_outa_z_score_singel_for_lable), 4),
             '3': np.round(get_mdd(bh_btc), 4),
             '4': np.round(get_mdd(bh_eth), 4),
             # '6': np.round(get_my_mdd(pt_outc), 4)
             }
            ]
table2 = pd.DataFrame(results2)
print_table(table2.values.tolist())

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

# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Open_R(-1)', speed_BTC_Open_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Open_R(-2)', speed_BTC_Open_R.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_High_R(-1)', speed_BTC_High_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_High_R(-2)', speed_BTC_High_R.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Low_R(-1)', speed_BTC_Low_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Low_R(-2)', speed_BTC_Low_R.shift(2))
training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Adj_R(-1)', speed_BTC_Adj_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Adj_R(-2)', speed_BTC_Adj_R.shift(2))

training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Close_R(-1)', speed_BTC_Close_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Close_R(-2)', speed_BTC_Close_R.shift(2))
training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Volume_R(-1)', speed_BTC_Volume_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_BTC_Volume_R(-2)', speed_BTC_Volume_R.shift(2))

speed_ETH_Open_R=speed_pair_feature['ETH_Open_R']
speed_ETH_High_R=speed_pair_feature['ETH_High_R']
speed_ETH_Low_R=speed_pair_feature['ETH_Low_R']
speed_ETH_Close_R=speed_pair_feature['ETH_Close_R']
speed_ETH_Volume_R=speed_pair_feature['ETH_Volume_R']
speed_ETH_Adj_R=speed_pair_feature['ETH_Ajd_R']

# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Open_R(-1)', speed_ETH_Open_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Open_R(-2)', speed_ETH_Open_R.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_High_R(-1)', speed_ETH_High_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_High_R(-2)', speed_ETH_High_R.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Low_R(-1)', speed_ETH_Low_R.shift(1))
# training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Low_R(-2)', speed_ETH_Low_R.shift(2))
training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Adj_R(-1)', speed_ETH_Adj_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Adj_R(-2)', speed_ETH_Adj_R.shift(2))

training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Close_R(-1)', speed_ETH_Close_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Close_R(-2)', speed_ETH_Close_R.shift(2))
training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Volume_R(-1)', speed_ETH_Volume_R.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_ETH_Volume_R(-2)', speed_ETH_Volume_R.shift(2))


# Insert into the trading data set
training_dataset.insert(len(training_dataset.columns), 'rbtc_ret(-1)', rbtc_ret.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rbtc_ret(-2)', rbtc_ret.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'rbtc_ret(-3)', rbtc_ret.shift(3))

training_dataset.insert(len(training_dataset.columns), 'reth_ret(-1)', reth_ret.shift(1))
training_dataset.insert(len(training_dataset.columns), 'reth_ret(-2)', reth_ret.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'reth_ret(-3)', reth_ret.shift(3))

training_dataset.insert(len(training_dataset.columns), 'rrbtc(-1)', rrbtc.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rrbtc(-2)', rrbtc.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'rrbtc(-3)', rrbtc.shift(3))

training_dataset.insert(len(training_dataset.columns), 'rreth(-1)', rreth.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rreth(-2)', rreth.shift(2))
# training_dataset.insert(len(training_dataset.columns), 'rreth(-3)', rreth.shift(3))
# The returns generated by pait trading and logarithmic returns
# training_dataset.insert(len(training_dataset.columns), 'Log_R(-1)', pair_trading_dataset['Log_R(-1)'])
# training_dataset.insert(len(training_dataset.columns), 'port_out(-1)', pair_trading_dataset['port_out(-1)'])
# training_dataset.insert(len(training_dataset.columns), 'Log_R(-2)', pair_trading_dataset['Log_R(-2)'])
# training_dataset.insert(len(training_dataset.columns), 'port_out(-2)', pair_trading_dataset['port_out(-2)'])

# training_dataset.insert(len(training_dataset.columns), 'up_th(-1)', pair_trading_dataset['up_th'].shift(1))
# training_dataset.insert(len(training_dataset.columns), 'lw_th(-1)', pair_trading_dataset['lw_th'].shift(1))
#
# training_dataset.insert(len(training_dataset.columns), 'up_th(-2)', pair_trading_dataset['up_th'].shift(2))
# training_dataset.insert(len(training_dataset.columns), 'lw_th(-2)', pair_trading_dataset['lw_th'].shift(2))
#
# training_dataset.insert(len(training_dataset.columns), 'up_th(-3)', pair_trading_dataset['up_th'].shift(3))
# training_dataset.insert(len(training_dataset.columns), 'lw_th(-3)', pair_trading_dataset['lw_th'].shift(3))
# training_dataset.insert(len(training_dataset.columns), 'pair_trading_signal', pair_trading_dataset['ftestsig2'])
#
slide_window=3
up_th = (z_score.rolling(window=slide_window,min_periods=1).mean()) + (z_score.rolling(window=slide_window,min_periods=1).std() * (a-1))  # upper threshold
lw_th = (z_score.rolling(window=slide_window,min_periods=1).mean()) - (z_score.rolling(window=slide_window,min_periods=1).std() * (1-b))  # lower threshold
up_th = up_th.dropna()
lw_th = lw_th.dropna()

rate_up_th=up_th.pct_change(1).dropna()
rate_lw_th=lw_th.pct_change(1).dropna()
training_dataset.insert(len(training_dataset.columns), 'up_th(-1)', rate_up_th.shift(1))
training_dataset.insert(len(training_dataset.columns), 'lw_th(-1)', rate_lw_th.shift(1))

training_dataset.insert(len(training_dataset.columns), 'up_th(-2)', rate_up_th.shift(2))
training_dataset.insert(len(training_dataset.columns), 'lw_th(-2)', rate_lw_th.shift(2))

training_dataset.insert(len(training_dataset.columns), 'up_th(-3)', rate_up_th.shift(3))
training_dataset.insert(len(training_dataset.columns), 'lw_th(-3)', rate_up_th.shift(3))



speed_z_score=(z_score.pct_change(1).dropna()).pct_change(1).dropna()
training_dataset.insert(len(training_dataset.columns), 'speed_z_score(-1)', speed_z_score.shift(1))
training_dataset.insert(len(training_dataset.columns), 'speed_z_score(-2)', speed_z_score.shift(2))

rate_z_score=z_score.pct_change(1).dropna()
training_dataset.insert(len(training_dataset.columns), 'rate_z_score(-1)', rate_z_score.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rate_z_score(-2)', rate_z_score.shift(2))





spread_BTC_low_high=nor_pair_feature.BTC_High_R-nor_pair_feature.BTC_Low_R
rate_spread_BTC_low_high=(spread_BTC_low_high.pct_change(1).dropna()).pct_change(1).dropna()

spread_ETH_low_high=nor_pair_feature.ETH_High_R-nor_pair_feature.ETH_Low_R
rate_spread_ETH_low_high=(spread_ETH_low_high.pct_change(1).dropna()).pct_change(1).dropna()

training_dataset.insert(len(training_dataset.columns), 'rate_spread_BTC_low_high(-1)', rate_spread_BTC_low_high.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rate_spread_BTC_low_high(-2)', rate_spread_BTC_low_high.shift(2))
training_dataset.insert(len(training_dataset.columns), 'rate_spread_BTC_low_high(-3)', rate_spread_BTC_low_high.shift(3))

training_dataset.insert(len(training_dataset.columns), 'rate_spread_ETH_low_high(-1)', rate_spread_ETH_low_high.shift(1))
training_dataset.insert(len(training_dataset.columns), 'rate_spread_ETH_low_high(-2)', rate_spread_ETH_low_high.shift(2))
training_dataset.insert(len(training_dataset.columns), 'rate_spread_ETH_low_high(-3)', rate_spread_ETH_low_high.shift(3))



training_dataset=training_dataset.dropna()
training_dataset.to_csv("../4.Generate y_predict using Adaboost (traing dataset$testing dataset)/0617_training_dataset.csv", index=True)

print('08311518')