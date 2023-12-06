import os
from config import Config
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import Trials
from hyperopt import fmin
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import datetime
import pandas as pd
from numpy import array
from sklearn.metrics import roc_curve,accuracy_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from optml.genetic_optimizer import GeneticOptimizer
from optml.optimizer_base import Parameter
from optml.hyperopt_optimizer import HyperoptOptimizer
from catboost import CatBoostClassifier
import sklearn
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report



def lgbm_opt(X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=Config.KFOLD,shuffle=True, random_state=Config.SEED)

    def roc_auc_cv(params, random_state=Config.SEED, cv=kf, X=X_train, y=y_train):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate']}
        model = lgb.LGBMClassifier(random_state=random_state, num_leaves=4, **params)
        score = -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score

    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
        'max_depth' : hp.quniform('max_depth', 4, 20, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0)}
    trials = Trials()
    best = fmin(
        fn=roc_auc_cv, space=space, algo=tpe.suggest,
        max_evals=Config.HYPEROPT_MAX_EVAL, trials=trials, rstate=np.random.RandomState(Config.SEED))
    clf = lgb.LGBMClassifier(
        random_state=Config.SEED, n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/LGBM_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def svm_opt(X_train, y_train, X_test, y_test):
    model = svm.SVC(random_state=Config.SEED, decision_function_shape='ovo')
    params = [
        Parameter(name='C', param_type='continuous', lower=0.01, upper=1.0)]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/SVM_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def sgd_opt(X_train, y_train, X_test, y_test):
    model = linear_model.SGDClassifier(random_state=Config.SEED)
    params = [
        Parameter(name='alpha', param_type='continuous', lower=0.01, upper=1.0)]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/SGD_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def knn_opt(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    # params = [
    #     Parameter(name='n_neighbors', param_type='integer', lower=3, upper=4)
    # ]
    # optimizer = HyperoptOptimizer(
    #     model=model,
    #     eval_func=accuracy_score,
    #     hyperparams=params)
    # _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # # Save model
    # folder_path = "./checkpoints"
    # if os.path.isdir(folder_path):
    #     pass
    # else:
    #     os.mkdir(folder_path)
    # file_name = "./checkpoints/KNN_{}.bin".format(datetime.datetime.today().date())
    # joblib.dump(clf, file_name)

    return y_pred

def gpc_opt(X_train, y_train, X_test, y_test):
    clf = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=Config.SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/GPC_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def gnb_opt(X_train, y_train, X_test, y_test):
    clf = naive_bayes.GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/GNB_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def dtc_opt(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=Config.SEED)   #OneVsRestClassifier(DecisionTreeClassifier())
    params = [
        Parameter(name='max_depth', param_type='integer', lower=3, upper=10),
        Parameter(name='min_samples_split', param_type='integer', lower=2, upper=5),
        Parameter(name='min_samples_leaf', param_type='continuous', lower=0, upper=0.5)]

    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/DTC_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def ada_opt(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(random_state=Config.SEED)
    params = [
        Parameter(name='n_estimators', param_type='integer', lower=100, upper=3000),
        Parameter(name='learning_rate', param_type='continuous', lower=0.0001, upper=0.5)
    ]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/ADA_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def gbc_opt(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=Config.SEED)
    params = [
        Parameter(name='n_estimators', param_type='integer', lower=50, upper=1000),
        Parameter(name='max_depth', param_type='integer', lower=3, upper=10),
        Parameter(name='learning_rate', param_type='continuous', lower=0.001, upper=0.5),
        Parameter(name='min_samples_split', param_type='continuous', lower=0, upper=0.5),
        Parameter(name='min_samples_leaf', param_type='continuous', lower=0, upper=0.5)]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/GBC_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred


def lgbm_opt(X_train, y_train, X_test, y_test):
    # kf = KFold(n_splits=Config.KFOLD, random_state=Config.SEED)
    kf = KFold(n_splits=Config.KFOLD, shuffle=True, random_state=Config.SEED)

    def roc_auc_cv(params, random_state=Config.SEED, cv=kf, X=X_train, y=y_train):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate']}
        model = lgb.LGBMClassifier(random_state=random_state, num_leaves=4, **params)
        score = -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
        'max_depth' : hp.quniform('max_depth', 4, 20, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0)}
    trials = Trials()
    best = fmin(
        fn=roc_auc_cv, space=space, algo=tpe.suggest,
        max_evals=Config.HYPEROPT_MAX_EVAL, trials=trials, rstate=np.random.RandomState(Config.SEED))
    clf = lgb.LGBMClassifier(
        random_state=Config.SEED, n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/LGBM_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def xgb_opt(X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=Config.KFOLD, shuffle=True, random_state=Config.SEED)

    def roc_auc_cv(params, random_state=Config.SEED, cv=kf, X=X_train, y=y_train):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'min_child_weight': params['min_child_weight'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'subsample': params['subsample']}
        model = xgb.XGBClassifier(random_state=random_state, **params)
        score = -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score

    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
        'max_depth' : hp.quniform('max_depth', 4, 20, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1, 0.05),
        'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 10, 0.05)}
    trials = Trials()
    best = fmin(
        fn=roc_auc_cv, space=space, algo=tpe.suggest,
        max_evals=Config.HYPEROPT_MAX_EVAL, trials=trials, rstate=np.random.RandomState(Config.SEED))
    clf = xgb.XGBClassifier(
        random_state=Config.SEED, n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),learning_rate=best['learning_rate'],
        min_child_weight=best['min_child_weight'], colsample_bytree=best['colsample_bytree'],
        gamma=best['gamma'], subsample=best['subsample'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/XGB_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def cat_opt(X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=Config.KFOLD, shuffle=True, random_state=Config.SEED)

    def roc_auc_cv(params, random_state=Config.SEED, cv=kf, X=X_train, y=y_train):
        params = {
            'depth': int(params['depth']),
            'learning_rate': params['learning_rate']}
        model = CatBoostClassifier(iterations=10, random_seed=random_state, silent=True, **params)
        score = -cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        return score

    space = {
        'depth': hp.quniform('depth', 4, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -4, 0)}
    trials = Trials()
    best = fmin(
        fn=roc_auc_cv, space=space, algo=tpe.suggest,
        max_evals=Config.HYPEROPT_MAX_EVAL, trials=trials, rstate=np.random.RandomState(Config.SEED))
    clf = CatBoostClassifier(
        iterations=10, random_state=Config.SEED, silent=True,
        depth=int(best['depth']),learning_rate=best['learning_rate'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/CAT_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def ridge_opt(X_train, y_train, X_test, y_test):
    model = linear_model.RidgeClassifier(random_state=Config.SEED)
    params = [
        Parameter(name='alpha', param_type='continuous', lower=0.01, upper=1.0)]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/RIDGE_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred

def mlp_opt(X_train, y_train, X_test, y_test):
    model = linear_model.Perceptron(random_state=Config.SEED)
    params = [
        Parameter(name='alpha', param_type='continuous', lower=0.01, upper=1.0)]
    optimizer = HyperoptOptimizer(
        model=model,
        eval_func=accuracy_score,
        hyperparams=params)
    _, clf = optimizer.fit(X_train, y_train, n_iters=Config.HYPEROPT_MAX_EVAL)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    # Save model
    folder_path = "./checkpoints"
    if os.path.isdir(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    file_name = "./checkpoints/MLP_{}.bin".format(datetime.datetime.today().date())
    joblib.dump(clf, file_name)

    return clf, y_pred


def get_data_raw(testing_start_index,drop_columns):

    total_dataset = "0902_training_dataset.csv"
    dataset_df = pd.read_csv(total_dataset, parse_dates=[0], index_col=0)
    dataset_df = dataset_df.dropna()

    X = dataset_df.drop(columns=drop_columns) #, 'port_outa_z_score_singel_for_lable'])   ,'Log_R(-1)','port_out(-1)','Log_R(-2)','port_out(-2)'
    y = dataset_df['z_score_singel_for_lable']

    X=X.loc[:'2022-04-01', ]
    y=y.loc[:'2022-04-01', ]

    class_to_index = {0: 0, 1: 1, -1: 2}

    y = [class_to_index[label] for label in y]

    return X, y

# split a multivariate sequence into samples
def split_sequences(input_sequences, output, n_steps_in):
    X, y = list(), list()
    for i in range(len(input_sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix > len(input_sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def Mlp_opt(X_train, y_train, X_test, y_test):
    # Create an instance of MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(300,), activation='tanh', solver='sgd',learning_rate='adaptive', random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    if Config.PRINT_CLASSIFICATION_REPORT:
        print(classification_report(y_test, y_pred))

    return  y_pred


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out




def gbc_ada_opt_test(X_train, y_train, X_test):

    # Create a gradient lift tree as the base classifier for AdaBoostClassifier
    base_classifier = GradientBoostingClassifier(n_estimators=190, max_depth=9, random_state=1016,verbose=3)
    # Create an AdaBoostClassifier and use the gradient lift tree as the base classifier
    adaboost = AdaBoostClassifier(estimator=base_classifier,learning_rate=0.0001, n_estimators=190, random_state=1016)
    # Fit the AdaBoostClassifier model on the training set
    adaboost.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = adaboost.predict(X_test)
    return  y_pred

#
# from hyperopt import hp, tpe
# from optml.hyperopt_optimizer import HyperoptOptimizer
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# import numpy as np
#
# def gbc_ada_opt(X_train, y_train, X_test, y_test):
#     # 创建梯度提升树作为AdaBoostClassifier的基分类器
#     base_classifier = GradientBoostingClassifier(n_estimators=1100, max_depth=9, random_state=1016)
#     # 创建AdaBoostClassifier，并将梯度提升树作为基分类器
#     adaboost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=1100, random_state=1016)
#
#     # 定义参数搜索空间
#     space = {
#         'n_estimators': hp.quniform('n_estimators', 100, 3000, 1),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5))
#     }
#
#     # 定义目标函数
#     def objective(params):
#         n_estimators = int(params['n_estimators'])
#         learning_rate = params['learning_rate']
#
#         # 使用传递的模型对象
#         adaboost.base_estimator.n_estimators = n_estimators
#         adaboost.learning_rate = learning_rate
#
#         # 使用交叉验证评估模型性能
#         scores = cross_val_score(adaboost, X_train, y_train, cv=5, scoring='accuracy')
#         avg_score = np.mean(scores)
#
#         return -avg_score  # hyperopt使用最小化目标函数，因此取负号
#
#     # 使用HyperoptOptimizer进行参数优化
#     optimizer = HyperoptOptimizer(model=adaboost, eval_func=objective, hyperparams=space, algo=tpe.suggest)
#     best_params, _ = optimizer.fit(X_train, y_train, n_iters=100)
#
#     # 使用最佳参数重新训练模型
#     n_estimators = int(best_params['n_estimators'])
#     learning_rate = best_params['learning_rate']
#     adaboost.base_estimator.n_estimators = n_estimators
#     adaboost.learning_rate = learning_rate
#     adaboost.fit(X_train, y_train)
#
#     # 在测试集上进行预测
#     y_pred = adaboost.predict(X_test)
#
#     return y_pred
