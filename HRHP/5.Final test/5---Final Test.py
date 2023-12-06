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
from result_util import get_pairstrategy_return_for_test

# Considering the consistency of the data (moving average window), the entire data set is used
testing_start_index = '2022-09-01'
testing_end_index = '2023-12-01'

BTC = yf.download('BTC-USD', start=testing_start_index, end=testing_end_index) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=testing_start_index, end=testing_end_index)  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)

pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()



pair_feature = pd.concat([BTC[['Open', 'High', 'Low', 'Close', 'Volume']], ETH[['Open', 'High', 'Low', 'Close', 'Volume']]], ignore_index=True, axis=1)
pair_feature.columns=['BTC_Open_R', 'BTC_High_R', 'BTC_Low_R', 'BTC_Close_R', 'BTC_Volume_R', 'ETH_Open_R', 'ETH_High_R', 'ETH_Low_R', 'ETH_Close_R', 'ETH_Volume_R']

pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()

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

tests_for_lable= pd.concat([btc_R ,eth_R], ignore_index=False,axis=1)
tests_for_lable = tests_for_lable.dropna()
# -------------------------------The hyperparameter obtained by GA----------------
a = 1.1730277004674885  #Return  920.9439
b = 0.8829080292183107
k = 2
window1 = 1
window2 = 79

# a = 1.600633084033785  #Return  80.3575
# b = 0.43041683982038315
# k = 4
# window1 = 4
# window2 = 78

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

# a = 1.1937296904781687  #Return : 18.7285
# b = 0.05222374794788317
# k = 2
# window1 = 23
# window2 =81

# a = 1.1486201606191768  #Return : 76.6564  Max Drawdown: -0.4982
# b = 0.30927478623887855
# k = 5
# window1 =19
# window2 = 86


# -------------------------------The hyperparameter obtained by GA----------------
# btc_R ,eth_R
# Generate a z-score based on the above hyperparameters
z_score,_ = get_z_socre_two_windows(btc_R ,eth_R,window1,window2)
z_score = z_score.dropna()
tests_for_lable.insert(len(tests_for_lable.columns), 'rbtc_ret', rbtc_ret)
tests_for_lable.insert(len(tests_for_lable.columns), 'reth_ret', reth_ret)
tests_for_lable.insert(len(tests_for_lable.columns), 'z_score', z_score)

# Set test period
# testing_start_index='2022-04-01'
# testing_end_index='2023-06-16'

final_test_dataset=tests_for_lable.loc[testing_start_index:testing_end_index, ['rbtc_ret','reth_ret','z_score']]


# ------------------------------triple_barrier Generated labels, for testing purposes, not important--------------------
# z_score_ret = triple_barrier(z_score, a, b, k)
# z_score_singel_for_lable = z_score_ret['triple_barrier_signal']
# final_test_dataset.insert(len(final_test_dataset.columns), 'z_score_singel_for_lable', z_score_singel_for_lable)
# port_out_for_lable = 0.0
# port_outa_for_lable = []
# for i in range(0, len(final_test_dataset.index)):
#     if final_test_dataset.at[final_test_dataset.index[i], 'z_score_singel_for_lable'] == 1:
#         '''
#         If the value of the z-score touches the upper threshold,
#         indicating a positive deviation from the mean,
#         it means that the growth rate of BTC is too fast. Therefore,
#         it is recommended to buy ETH.
#         '''
#         port_out_for_lable = final_test_dataset.at[final_test_dataset.index[i], 'reth_ret']
#     elif final_test_dataset.at[final_test_dataset.index[i], 'z_score_singel_for_lable'] == -1:
#         '''
#         If the value of z_score touches the lower barrier,
#         indicating a negative deviation from the mean,
#         it means that the growth rate of ETH is too fast.
#         Therefore, buy BTC.
#         '''
#         port_out_for_lable = final_test_dataset.at[final_test_dataset.index[i], 'rbtc_ret']
#     else:
#         port_out_for_lable = 0
#     port_outa_for_lable.append(port_out_for_lable)
# final_test_dataset.insert(len(final_test_dataset.columns), 'port_outa_for_lable', port_outa_for_lable)
# print("-------Lableing method---------Lableing-----------Lableing------------------------")
# port_outa_for_lable = (1 + final_test_dataset['port_outa_for_lable']).cumprod() # pair trading return
# MDD = get_mdd(port_outa_for_lable)
# print("Return : " + str(np.round(port_outa_for_lable.iloc[-1], 4)))
# print("Standard Deviation : " + str(
#     np.round(np.std(port_outa_for_lable), 4)))  # mean_absolute_percentage_error
# print("Sharpe Ratio (Rf=0%) : " + str(
#     np.round(port_outa_for_lable.iloc[-1] / (np.std(port_outa_for_lable)), 4)))
# print("Max Drawdown: " + str(np.round(MDD, 4)))  # calculate_mdd(pt_out)
# print("-------Lableing method---------Lableing-----------Lableing------------------------")
# ------------------------------triple_barrier Generated tags, for testing purposes, not important------------------------------

# ----------------------------------------import the Adabost prediction labels, very important----------------------------------------
final_label_path = "0905_y_prediction_HPHR.csv"#"0829_y_prediction_LPLR.csv"#"0829_y_prediction_HPHR.csv"#"0618_y_prediction_HPHR.csv" # "0618_y_prediction_lPLR.csv" "0829_y_prediction_LPLR.csv"
final_label = pd.read_csv(final_label_path, parse_dates=[0], index_col=0)
final_test_dataset.insert(len(final_test_dataset.columns), 'predict_final_label', final_label['y_pred'])
predict_port_out_for_lable = 0.0
predict_port_outa_for_lable = []
for i in range(0, len(final_test_dataset.index)):
    if final_test_dataset.at[final_test_dataset.index[i], 'predict_final_label'] == 1:
        '''
        If the value of the z-score touches the upper threshold, 
        indicating a positive deviation from the mean, 
        it means that the growth rate of BTC is too fast. Therefore,
        it is recommended to buy ETH.
        '''
        predict_port_out_for_lable = final_test_dataset.at[final_test_dataset.index[i], 'reth_ret']
    elif final_test_dataset.at[final_test_dataset.index[i], 'predict_final_label'] == 2:#   (class_to_index = {0: 0, 1: 1, -1: 2})
        '''
        If the value of z_score touches the lower barrier, 
        indicating a negative deviation from the mean, 
        it means that the growth rate of ETH is too fast. 
        Therefore, buy BTC.
        '''
        predict_port_out_for_lable = final_test_dataset.at[final_test_dataset.index[i], 'rbtc_ret']
    else:
        predict_port_out_for_lable = 0
    predict_port_outa_for_lable.append(predict_port_out_for_lable)  #daily return

final_test_dataset.insert(len(final_test_dataset.columns), 'predict_port_outa_for_lable', predict_port_outa_for_lable)
final_test_dataset = final_test_dataset.fillna(method='ffill')

print("--final result-----adaboost prediction very important-----------------------")
predict_port_outa_for_lable= (1 + final_test_dataset['predict_port_outa_for_lable']).cumprod()
predict_MDD = get_mdd(predict_port_outa_for_lable)
print("Return : " + str(np.round(predict_port_outa_for_lable.iloc[-1], 4)))
print("Standard Deviation : " + str(
    np.round(np.std(predict_port_outa_for_lable), 4)))  # mean_absolute_percentage_error
print("Sharpe Ratio (Rf=0%) : " + str(
    np.round(predict_port_outa_for_lable.iloc[-1] / (np.std(predict_port_outa_for_lable)), 4)))
print("Max Drawdown: " + str(np.round(predict_MDD, 4)))  # calculate_mdd(pt_out)
print("--final result-----adaboost prediction very important-----------------------")
# ----------------------------------------import the Adabost prediction tag, very important----------------------------------------



# ------------------------buy and hold strategy------------------------
bh_btc_test= (1 + final_test_dataset['rbtc_ret']).cumprod()
bh_eth_test= (1 + final_test_dataset['reth_ret']).cumprod()
# ------------------------buy and hold strategy------------------------


# ------------------------pair trading strategy------------------------
pt_out_pair_trading,_=get_pairstrategy_return_for_test(testing_start_index,testing_end_index)
# pt_out_pair_trading=pt_out_pair_trading.loc[testing_start_index:testing_end_index]
port_outa_pair_trading = (1 + pt_out_pair_trading).cumprod() # pair trading return
# ------------------------pair trading strategy------------------------



plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(predict_port_outa_for_lable, label='Machine Learning Model Prediction',color='r')
# plt.plot(port_outa_for_lable, label='triple barrier method ',color='b')
plt.plot(port_outa_pair_trading, label='Cumulative return on Pair Trading Strategy',color='y')
plt.plot(bh_btc_test, label='Cumulative return on Buy and Hold Bitcoin',color='g')
plt.plot(bh_eth_test, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
plt.title('Cumulative Returns of Machine Learning Model Predicted Trading Signals')
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.suptitle('Cumulative Returns')
ax.legend(loc='best')
ax.grid(True)
plt.show()



results2 = [{'0': 'Test:',
             '1': 'P-Trading Strategy',
             '2': 'Machine Learning Prediction',
             '3': 'Buy&Hold Bitcoin',
             '4': 'Buy&Hold Ethereum',},

            {'0': 'Return',
             '1': np.round(port_outa_pair_trading.iloc[-1], 4),
             '2': np.round(predict_port_outa_for_lable.iloc[-1], 4),
             '3': np.round(bh_btc_test.iloc[-1], 4),
             '4': np.round(bh_eth_test.iloc[-1], 4),
             # '6': np.round(pt_outc.iloc[-1], 4)
             },

            {'0': 'Standard Deviation',
             '1': np.round(np.std(port_outa_pair_trading), 4),
             '2': np.round(np.std(predict_port_outa_for_lable), 4),
             '3': np.round(np.std(bh_btc_test), 4),
             '4': np.round(np.std(bh_eth_test), 4),
             # '6': np.round(np.std(pt_outc), 4)
             },

            {'0': 'Sharpe Ratio (Rf=0%)',
             '1': np.round(port_outa_pair_trading.iloc[-1] / (np.std(port_outa_pair_trading)), 6),
             '2': np.round(predict_port_outa_for_lable.iloc[-1] / (np.std(predict_port_outa_for_lable)), 6),
             '3': np.round(bh_btc_test.iloc[-1] / (np.std(bh_btc_test)), 6),
             '4': np.round(bh_eth_test.iloc[-1] / (np.std(bh_eth_test)), 6),
             # '6': np.round(pt_outc.iloc[-1] / (np.std(pt_outc)), 4)
             },

            {'0': 'Max Drawdown',
             '1': np.round(get_mdd(port_outa_pair_trading), 6),
             '2': np.round(get_mdd(predict_port_outa_for_lable), 6),
             '3': np.round(get_mdd(bh_btc_test), 6),
             '4': np.round(get_mdd(bh_eth_test), 6),
             # '6': np.round(get_my_mdd(pt_outc), 4)
             }
            ]


table2 = pd.DataFrame(results2)
print_table(table2.values.tolist())


print('08311518')