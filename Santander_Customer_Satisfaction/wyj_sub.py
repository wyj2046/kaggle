# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def get_remove_col(train):
    remove = []
    # 去除常数列
    for c in train.columns:
        if train[c].std() == 0:
            remove.append(c)
    # 去除重复的列
    columns = train.columns
    for i in range(len(columns) - 1):
        v = train[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(v, train[columns[j]].values):
                remove.append(columns[j])
    return remove


def get_pred_y1(train_X, train_y, test_X):
    xg_train_X = xgb.DMatrix(train_X.values, label=train_y.values, feature_names=train_X.columns.tolist())
    xg_test_X = xgb.DMatrix(test_X.values, feature_names=test_X.columns.tolist())

    param = {}
    param['nthread'] = 2
    param['silent'] = 1
    param['eta'] = 0.1
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['seed'] = 229

    # cv_result = xgb.cv(param, xg_train_X, num_boost_round=1000, nfold=3, metrics='auc', early_stopping_rounds=50, verbose_eval=True, show_stdv=False)

    watchlist = [(xg_train_X, 'train')]
    # num_round = cv_result.shape[0]
    num_round = 57
    print 'num_round', num_round
    bst = xgb.train(param, xg_train_X, num_round, watchlist)

    pred_y = bst.predict(xg_test_X)
    return pred_y


def get_pred_y2(train_X, train_y, test_X):
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=100, random_state=229, verbose=1)
    clf.fit(train_X, train_y)
    pred_y = clf.predict_proba(test_X)
    return pred_y[:, 1]


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    remove = get_remove_col(train)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    train_X = train.drop(['ID', 'TARGET'], axis=1)
    train_y = train['TARGET']
    test_X = test.drop(['ID'], axis=1)

    pred_y1 = get_pred_y1(train_X, train_y, test_X)
    # pred_y2 = get_pred_y2(train_X, train_y, test_X)

    # pred_y = (pred_y1 * 4 + pred_y2 * 1) / 5
    pred_y = pred_y1

    submission = pd.DataFrame({"ID": test['ID'], 'TARGET': pred_y})
    columns = ['ID', 'TARGET']
    submission.to_csv('result.csv', index=False, columns=columns)
