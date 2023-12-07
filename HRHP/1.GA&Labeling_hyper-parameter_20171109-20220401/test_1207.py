# univariate mlp example

from mealpy.evolutionary_based import GA
# import tensorflow as tf
import statsmodels.regression.linear_model as rg
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

class HybridMlp(Problem):
    def __init__(self,dataset,GA_epoch,pop_size):  #dataset,, pair_train
        self.dataset = dataset
        # self.pair_train = pair_train
        self.GA_epoch = GA_epoch
        self.pop_size = pop_size

    def __init__(self, lb, ub, minmax, data=None, name="TimeSeries Multi-Layer Perceptron", **kwargs):
        super().__init__(lb, ub, minmax, data=data,
                         **kwargs)  ## data is needed because when initialize the Problem class, we need to check the output of fitness
        self.data = data
        self.name = name


    def create_problem(self):

        LB = [1.01,    0.01,   2  ,  1,   20,  0      ]
        UB = [1.8,     0.99,   60 ,  30,  90,  2.99   ]

        self.problem = {
            "fit_func": self.fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "max",
            "log_to": None,
            "obj_weights": [0.40,0.60],
            "save_population": False,
        }

    def decode_solution(self,solution):
        a = solution[0]
        b = solution[1]
        k = int(solution[2])
        window1= int(solution[3])
        window2 = int(solution[4])
        function_integer = int(solution[5])

        return {
            "a": a,
            "b": b,
            "k": k,
            "window1":window1,
            "window2": window2,
            "function_integer":function_integer
        }
    def fitness_function(self, solution):

        structure = self.decode_solution(solution)

        a = structure["a"]      #upper threshold
        b = structure["b"]      #lower threshold
        k=  structure["k"]      #max holding period
        window1 = structure["window1"]      #moving window 1
        window2 = structure["window2"]      #moving window 2
        function_integer=structure["function_integer"]      # calculation of Z-score

        tests = self.dataset.dropna()

        function_dict = {
            0: get_z_socre_hege,
            1: get_z_socre_no_hege,
            2: get_z_socre_two_windows
        }

        z_score_function = function_dict[function_integer]

        z_score,function_name = z_score_function(rbtc_ret,reth_ret,window1,window2)

        z_score = z_score.dropna()

        z_score_ret = triple_barrier(z_score, a, b, k)

        z_score_singel = z_score_ret['triple_barrier_signal']

        tests.insert(len(tests.columns), 'rbtc_ret', rbtc_ret)  # daily return of BTC
        tests.insert(len(tests.columns), 'reth_ret', reth_ret)  # daily return of ETH
        tests.insert(len(tests.columns), 'z_score_singel', z_score_singel)


        # port_out_z_score_singel = 0.0
        port_outa_z_score_singel = []
        MDD=-1
        try :
            for i in range(0, len(tests.index)):
                if tests.at[tests.index[i], 'z_score_singel'] == 1:  #
                    '''
                    If the value of the z-score touches the upper threshold, 
                    indicating a positive deviation from the mean, 
                    it means that the growth rate of BTC is too fast. Therefore,
                    it is recommended to buy ETH.
                    '''
                    port_out_z_score_singel = tests.at[tests.index[i], 'reth_ret']
                elif tests.at[tests.index[i], 'z_score_singel'] == -1:
                    '''
                    If the value of z_score touches the lower barrier, 
                    indicating a negative deviation from the mean, 
                    it means that the growth rate of ETH is too fast. 
                    Therefore, buy BTC.
                    '''
                    port_out_z_score_singel = tests.at[tests.index[i], 'rbtc_ret']
                else:
                    port_out_z_score_singel = 0
                port_outa_z_score_singel.append(port_out_z_score_singel)

            tests.insert(len(tests.columns), 'port_outa_z_score_singel', port_outa_z_score_singel)
            tests = tests.fillna(method='ffill')

            port_outa_z_score_singel = (1 + tests['port_outa_z_score_singel']).cumprod() # accumulative return

            MDD = get_mdd(port_outa_z_score_singel)
            if abs(MDD) < 0.7 and np.round(port_outa_z_score_singel.iloc[-1], 4) > 40:
                # print("Condition met")
                print("------FINALLY---------------------------------------")
                print("Return : " + str(np.round(port_outa_z_score_singel.iloc[-1], 4)))
                print("Standard Deviation : " + str(np.round(np.std(port_outa_z_score_singel), 4)))
                print("Sharpe Ratio (Rf=0%) : " + str(np.round(port_outa_z_score_singel.iloc[-1] / (np.std(port_outa_z_score_singel)), 4)))
                print("Max Drawdown: " + str(np.round(MDD, 4)))  # calculate_mdd(pt_out)
                print('++++++++++++++++++++++++++++++++++++++')
                print("a : " + str(a))
                print("b : " + str(b))
                print("k : " + str(k))
                print("window1 : " + str(window1))
                print("window2 : " + str(window2))
                print("Z-score : "+ str(function_name))
                print('-------------------------------------')
        except Exception as e:
            print("except:", e)

        fitness=[np.round(port_outa_z_score_singel.iloc[-1], 4), np.round(MDD+10, 4)]

        return fitness
    def training(self):
        self.create_problem()
        self.model = GA.BaseGA(GAepoch=self.GA_epoch,pop_size=self.pop_size, pc=0.7, pm=0.4)
        self.solution, self.best_fit = self.model.solve(self.problem)

    def best_fitness(self):
        Return, MDD = self.best_fit
        print('-------------------------------------')
        print("Return : " + str(Return))
        print("Max Drawdown: " + str(MDD))
        print('-------------------------------------')

    def best_model(self):
        structure = self.decode_solution(self.solution)
        print("---------BEST---BEST--FINALLY-------BEST-------BEST---------------------")
        print("a", structure["a"], )
        print("b", structure["b"], )
        print("k",structure["k"],)
        print("window1", structure["window1"])
        print("window2 : ", structure["window2"])




import logging
def initLogging(logFilename):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

initLogging('test_infor.log')
logging.info('just play')

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger("1207.log", sys.stdout)
sys.stderr = Logger("test_error.log", sys.stderr)		# redirect std err, if necessary





GA_epoch=100
GA_pop_size=20


## Create hybrid model
model = HybridMlp(tests,GA_epoch,GA_pop_size) #dataset,

model.training()

model.best_model()

model.best_fitness()
