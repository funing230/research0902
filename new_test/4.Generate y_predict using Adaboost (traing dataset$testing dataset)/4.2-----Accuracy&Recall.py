import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util import get_data_raw,train_test_split,\
    lgbm_opt,svm_opt,sgd_opt,gpc_opt,gnb_opt,dtc_opt,ada_opt,gbc_opt,lgbm_opt\
    ,xgb_opt,cat_opt,ridge_opt,mlp_opt,Mlp_opt,knn_opt,gbc_ada_opt_test


#remove 'BTC_RET', 'ETH_RET','rbtc_ret','reth_ret','z_score', 'port_outa_z_score_singel_for_lable','z_score_singel_for_lable'




def get_data_raw_train_test():

    total_dataset = "0617_training_dataset.csv"
    dataset_df = pd.read_csv(total_dataset, parse_dates=[0], index_col=0)
    dataset_df = dataset_df.dropna()

    X = dataset_df.drop(columns=['BTC_RET', 'ETH_RET','rbtc_ret','reth_ret',
                                 'z_score', 'port_outa_z_score_singel_for_lable',
                                 'z_score_singel_for_lable'
                                 ]) #, 'port_outa_z_score_singel_for_lable'])   ,'Log_R(-1)','port_out(-1)','Log_R(-2)','port_out(-2)'
    y = dataset_df['z_score_singel_for_lable']

    class_to_index = {0: 0, 1: 1, -1: 2}

    y = [class_to_index[label] for label in y]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = get_data_raw_train_test()


# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)     #0.4850631578947368
# clf, y_pred=svm_opt(X_train, y_train, X_test, y_test)      #0.5013440860215054
# clf, y_pred=sgd_opt(X_train, y_train, X_test, y_test)     #0.5
# clf, y_pred=gpc_opt(X_train, y_train, X_test, y_test)     #0.5068576653459311
# clf, y_pred=gnb_opt(X_train, y_train, X_test, y_test)  # Accuracy: 0.5069306930693069 0.6273503216229589
# clf, y_pred=dtc_opt(X_train, y_train, X_test, y_test)    #0.6247026169706582
# clf, y_pred=ada_opt(X_train, y_train, X_test, y_test)     # Accuracy: 0.7683168316831683 best----------------------0.6837837837837838
# clf, y_pred=gbc_opt(X_train, y_train, X_test, y_test)       #best--------- Accuracy: 0.8198019801980198            0.5522606045212091
# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)      #0.4921721721721722
# clf, y_pred=xgb_opt(X_train, y_train, X_test, y_test)      #0.4318461538461538
clf, y_pred=cat_opt(X_train, y_train, X_test, y_test)      #0.5492172211350294
# clf, y_pred=ridge_opt(X_train, y_train, X_test, y_test)      # 0.4534368299521689
# clf, y_pred=mlp_opt(X_train, y_train, X_test, y_test)      # 0.5059306198716387
# y_pred=Mlp_opt(X_train, y_train, X_test, y_test)      #  Accuracy: 0.46534653465346537  0.502687164104487
# y_pred=knn_opt(X_train, y_train, X_test, y_test)    # Accuracy: 0.504950495049505



# y_pred=gbc_ada_opt_test(X_train, y_train, X_test)  #best--best--best--best--------- Accuracy: 0.8198019801980198



y_pred=pd.DataFrame(y_pred)
y_pred.columns = ['y_pred']
y_pred.index=X_test.index

# pd.DataFrame(y_pred).to_csv("../5.Final test/0618_y_prediction_HPHR.csv", index=True)


# #------------------------------------------------------

from sklearn.metrics import classification_report, accuracy_score

# 假设y_true和y_pred是真实标签和预测标签
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# 逐个打印指标
for label, metrics in report.items():
    if label == 'accuracy':
        print(f"{label.capitalize()}: {metrics}")
    else:
        print(f"Label: {label}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1-score: {metrics['f1-score']}")
        print(f"Support: {metrics['support']}")
        print()

print(f"Accuracy: {accuracy}")
