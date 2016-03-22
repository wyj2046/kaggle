# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


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
    train_X = train.drop(['ID', 'TARGET'], axis=1)
    train_y = train['TARGET']
    test_X = test.drop(['ID'], axis=1)

    pred_y1 = get_pred_y1(train_X, train_y, test_X)
    pred_y2 = get_pred_y2(train_X, train_y, test_X)

    pred_y = (pred_y1 * 4 + pred_y2 * 1) / 5

    submission = pd.DataFrame(data=pred_y, columns=['TARGET'])
    submission = submission.join(test['ID'])
    columns = ['ID', 'TARGET']
    submission.to_csv('result.csv', index=False, columns=columns)
