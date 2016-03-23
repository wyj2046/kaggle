# -*- coding: utf-8 -*-
import sys
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


random_seed = 229


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


def tune_xgb_param(X, y, xgbcv=False):
    base_param = {}
    base_param['nthread'] = 2
    base_param['silent'] = 1
    base_param['seed'] = random_seed
    base_param['objective'] = 'binary:logistic'
    # base_param['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)

    base_param['learning_rate'] = 0.1
    base_param['n_estimators'] = 70
    base_param['max_depth'] = 5
    base_param['min_child_weight'] = 9
    base_param['gamma'] = 0.23
    base_param['subsample'] = 0.7
    base_param['colsample_bytree'] = 0.8

    if xgbcv:
        xg_train = xgb.DMatrix(X, label=y)
        cv_result = xgb.cv(base_param, xg_train, num_boost_round=base_param['n_estimators'], nfold=3, metrics='auc', early_stopping_rounds=50, verbose_eval=1, show_stdv=False, seed=random_seed)
        base_param['n_estimators'] = cv_result.shape[0]

    tune_param = {}
    # tune_param['max_depth'] = range(1, 10, 2)
    # tune_param['min_child_weight'] = range(1, 10, 2)
    # tune_param['min_child_weight'] = range(9, 20, 2)

    # tune_param['gamma'] = [i / 10.0 for i in range(0, 5)]
    # tune_param['gamma'] = [i / 100.0 for i in range(15, 25, 2)]
    # tune_param['gamma'] = [i / 100.0 for i in range(23, 35, 2)]

    # tune_param['subsample'] = [i / 10.0 for i in range(6, 10)]
    # tune_param['colsample_bytree'] = [i / 10.0 for i in range(6, 10)]
    # tune_param['colsample_bytree'] = [i / 100.0 for i in range(75, 95)]

    model = xgb.XGBClassifier(**base_param)
    clf = GridSearchCV(model, tune_param, scoring='roc_auc', n_jobs=4, cv=3, verbose=2)
    clf.fit(X, y)
    for item in clf.grid_scores_:
        print item
    print 'BEST', clf.best_params_, clf.best_score_

    return clf.best_estimator_


def get_pred_y1(train_X, train_y, test_X):
    X_fit, X_eval, y_fit, y_eval = train_test_split(train_X, train_y, test_size=0.1, random_state=random_seed)

    xgb_model = tune_xgb_param(train_X, train_y, True)

    xgb_model.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric='auc', eval_set=[(X_fit, y_fit), (X_eval, y_eval)])

    pred_y = xgb_model.predict_proba(test_X)
    return pred_y[:, 1]


def get_pred_y2(train_X, train_y, test_X):
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=100, random_state=random_seed, verbose=1)
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

    # tune_xgb_param(train_X, train_y)
    # sys.exit(0)

    pred_y1 = get_pred_y1(train_X, train_y, test_X)
    # pred_y2 = get_pred_y2(train_X, train_y, test_X)

    # pred_y = (pred_y1 * 4 + pred_y2 * 1) / 5
    pred_y = pred_y1

    submission = pd.DataFrame({"ID": test['ID'], 'TARGET': pred_y})
    columns = ['ID', 'TARGET']
    submission.to_csv('result.csv', index=False, columns=columns)
