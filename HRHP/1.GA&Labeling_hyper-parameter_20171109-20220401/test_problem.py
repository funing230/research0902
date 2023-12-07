from sklearn.svm import SVC
# from permetrics.classification import ClassificationMetric
from mealpy.utils.problem import Problem
import numpy as np
import random
random.seed(7)
np.random.seed(42)
# tf.random.set_seed(116)
from GA_util import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd,triple_barrier_change_rate
from baseline_util import get_z_socre_hege,get_z_socre_no_hege,get_z_socre_two_windows
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from mealpy.utils.problem import Problem

class ClassificationSVC(Problem):
    def __init__(self, lb, ub, minmax, data=None, name="Support Vector Classification", **kwargs):
        super().__init__(lb, ub, minmax, data=data, **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name

    def decode_solution(self, solution):
        a = solution[0]
        b = solution[1]
        k = int(solution[2])
        window1 = int(solution[3])
        window2 = int(solution[4])
        function_integer = int(solution[5])

        return {
            "a": a,
            "b": b,
            "k": k,
            "window1": window1,
            "window2": window2,
            "function_integer": function_integer
        }

    def generate_trained_model(self, structure):
        # print('Trying to generate trained model...')
        model = SVC(C=structure["C"], kernel=structure["kernel"])
        model.fit(self.data["X_train"], self.data["y_train"])
        # print("Return model")
        return model

    def generate_loss_value(self, structure):
        model = self.generate_trained_model(structure)

        # We take the loss value of validation set as a fitness value for selecting the best model demonstrate prediction
        y_pred = model.predict(self.data["X_test"])

        evaluator = ClassificationMetric(self.data["y_test"], y_pred, decimal=6)
        loss = evaluator.accuracy_score(average="macro")
        return loss

    def fit_func(self, solution):
        structure = self.decode_solution(solution)
        fitness = self.generate_loss_value(structure)
        return fitness




traing_start_index = '2017-11-09'
traing_end_index = '2022-09-01'

# traing_start_index,traing_end_index

BTC = yf.download('BTC-USD', start=traing_start_index, end=traing_end_index) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=traing_start_index, end=traing_end_index)  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)

pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()

pair_ret=normalize_series(pair)

#remove first row with NAs
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']

tests= pd.concat([pair_ret['BTC_RET'] ,pair_ret['ETH_RET']], ignore_index=False,axis=1)

hege= rg.OLS(pair_ret['BTC_RET'] ,pair_ret['ETH_RET']).fit().params[0]
# hege=1
pair_train= pair_ret['BTC_RET'] - hege * pair_ret['ETH_RET']
# BTC_ETH Rolling Spread Z-Score Calculation

rbtc_ret= pair_ret['BTC_RET']
reth_ret= pair_ret['ETH_RET']
