from mealpy.utils.problem import Problem
from mealpy import FloatVar,GA
from GA_util import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd,triple_barrier_change_rate
from baseline_util import get_z_socre_hege,get_z_socre_no_hege,get_z_socre_two_windows,calculate_sharpe_ratio
import yfinance as yf
import numpy as np
import pandas as pd
import sys

class TBl_problem(Problem):
    def __init__(self, bounds,minmax, data=None, name="Triple barrier labeling method", **kwargs):
        ## data is needed because when initialize the Problem class, we need to check the output of fitness
        super().__init__(bounds, minmax, data=data, **kwargs)
        self.data = data
        self.name = name

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

    def obj_func(self, solution):
        structure = self.decode_solution(solution)
        a = structure["a"]      #upper threshold
        b = structure["b"]      #lower threshold
        k=  structure["k"]      #max holding period
        window1 = structure["window1"]      #moving window 1
        window2 = structure["window2"]      #moving window 2
        function_integer=structure["function_integer"]      # calculation of Z-score

        tests = self.data.dropna()
        rbtc_ret=tests['BTC_RET']
        reth_ret=tests['ETH_RET']

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
            # sharp_ratio=calculate_sharpe_ratio(tests['port_outa_z_score_singel'])
            if abs(MDD) < 0.4 and np.round(port_outa_z_score_singel.iloc[-1], 4) > 60:
            # if sharp_ratio>2.1 :
                # print("Condition met")
                print("------FINALLY---------------------------------------")
                print("Return : " + str(np.round(port_outa_z_score_singel.iloc[-1], 4)))
                print("Standard Deviation : " + str(np.round(np.std(port_outa_z_score_singel), 4)))
                # print("Sharpe Ratio (Rf=0%) : " + str(np.round(port_outa_z_score_singel.iloc[-1] / (np.std(port_outa_z_score_singel)), 4)))
                print("Sharpe Ratio (Rf=0%):" + str(calculate_sharpe_ratio(tests['port_outa_z_score_singel'], 0)))
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

        # fitness=[np.round(port_outa_z_score_singel.iloc[-1], 4), np.round(MDD+10, 4)]
        # fitness = calculate_sharpe_ratio(tests['port_outa_z_score_singel'])
        fitness = np.round(MDD + 1, 4)
        return fitness


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger("1209MDD.log", sys.stdout)
sys.stderr = Logger("test_error1209MDD.log", sys.stderr)		# redirect std err, if necessary





traing_start_index = '2017-11-09'
traing_end_index = '2022-09-01'
BTC = yf.download('BTC-USD', start=traing_start_index, end=traing_end_index) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=traing_start_index, end=traing_end_index)  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)
pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()

pair_ret=normalize_series(pair)
pair_ret=pair_ret.tail(len(pair_ret)-1)

pair_ret.columns = ['BTC_RET','ETH_RET']
tests= pd.concat([pair_ret['BTC_RET'] ,pair_ret['ETH_RET']], ignore_index=False,axis=1)

LB = [1.01,    0.01,   2  ,  1,   20,  0      ]
UB = [1.8,     0.99,   60 ,  30,  90,  2.99   ]
problem = TBl_problem(bounds=FloatVar(lb=LB, ub=UB), minmax="max", data=tests, save_population=False, log_to=None) #obj_weights=[0.40, 0.60],

model = GA.BaseGA(epoch=2000, pop_size=200)
best_agent = model.solve(problem)

print(f"Best agent: {best_agent}")
print(f"Best solution: {best_agent.solution}")
print(f"Best accuracy: {best_agent.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(best_agent.solution)}")
