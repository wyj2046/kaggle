# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_X = train.drop(['ID', 'TARGET'], axis=1)
    train_y = train['TARGET']
    test_X = test.drop(['ID'], axis=1)

    xg_train_X = xgb.DMatrix(train_X.values, label=train_y.values, feature_names=train_X.columns.tolist())
    xg_test_X = xgb.DMatrix(test_X.values, feature_names=test_X.columns.tolist())

    param = {}
    param['nthread'] = 2
    param['silent'] = 1
    param['eta'] = 0.1
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['seed'] = 229

    watchlist = [(xg_train_X, 'train')]
    num_round = 1000
    bst = xgb.train(param, xg_train_X, num_round, watchlist)

    pred_y = bst.predict(xg_test_X)

    submission = pd.DataFrame(data=pred_y, columns=['TARGET'])
    submission = submission.join(test['ID'])
    columns = ['ID', 'TARGET']
    submission.to_csv('result.csv', index=False, columns=columns)
