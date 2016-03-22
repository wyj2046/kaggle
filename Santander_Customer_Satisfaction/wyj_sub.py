# -*- coding: utf-8 -*-
import sys
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


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


def model_fit(xgb_model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=229)
    xgb_model.fit(X, y, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)])


def tune_xgb_param(X, y):
    base_param = {}
    base_param['nthread'] = 2
    base_param['silent'] = 1
    base_param['learning_rate'] = 0.1
    base_param['n_estimators'] = 57
    base_param['objective'] = 'binary:logistic'
    base_param['seed'] = 229
    model = xgb.XGBClassifier(**base_param)

    tune_param = {}
    tune_param['max_depth'] = range(3, 10, 2)
    tune_param['min_child_weight'] = range(1, 6, 2)

    clf = GridSearchCV(model, tune_param, scoring='roc_auc', n_jobs=2, cv=3, verbose=2)
    clf.fit(X, y)
    print clf.grid_scores_
    print clf.best_params_, clf.best_score_

    model_fit(clf.best_estimator_, X, y)


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

    tune_xgb_param(train_X, train_y)
    sys.exit(0)

    pred_y1 = get_pred_y1(train_X, train_y, test_X)
    # pred_y2 = get_pred_y2(train_X, train_y, test_X)

    # pred_y = (pred_y1 * 4 + pred_y2 * 1) / 5
    pred_y = pred_y1

    submission = pd.DataFrame({"ID": test['ID'], 'TARGET': pred_y})
    columns = ['ID', 'TARGET']
    submission.to_csv('result.csv', index=False, columns=columns)
