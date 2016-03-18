# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
from sklearn import metrics


def dates_to_seconds(dates):
    times = dates.split()[1]
    times_list = times.split(':')
    seconds = int(times_list[0]) * 3600 + int(times_list[1]) * 60 + int(times_list[2])
    return seconds


def harmonize_data2(df):
    data = pd.DataFrame(index=range(len(df)))
    data = df.get(['X', 'Y'])

    dows = df['DayOfWeek'].unique()
    dow_map = {}
    i = 0
    for item in dows:
        dow_map[item] = i
        i += 1
    data = data.join(df['DayOfWeek'].map(dow_map))

    pds = df['PdDistrict'].unique()
    pd_map = {}
    i = 0
    for item in pds:
        pd_map[item] = i
        i += 1
    data = data.join(df['PdDistrict'].map(pd_map))

    data['Seconds'] = df['Dates'].apply(dates_to_seconds)
    data['Hour'] = pd.to_datetime(df['Dates']).dt.hour
    data['Day'] = pd.to_datetime(df['Dates']).dt.day

    data['StreetCorner'] = df['Address'].str.contains('/').map(int)
    data['Block'] = df['Address'].str.contains('Block').map(int)

    return data


def label_to_num(df):
    labels = df.unique()
    label_map = {}
    i = 0
    for label in labels:
        label_map[label] = i
        i += 1
    df = df.map(label_map)
    return df


def model_fit(xgb_model, X, y, cv_folds=5, early_stopping_rounds=50):
    xgb_param = xgb_model.get_xgb_params()
    xgb_param['num_class'] = len(y.unique())
    xg_train = xgb.DMatrix(X.values, label=y.values)
    cv_result = xgb.cv(xgb_param, xg_train, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    print 'cv_result.shape[0]', cv_result.shape[0]
    xgb_model.set_params(n_estimators=cv_result.shape[0])

    xgb_model.fit(X, y, eval_metric='mlogloss')
    y_pred = xgb_model.predict_proba(X)

    print 'logloss: %f' % metrics.log_loss(y, y_pred)


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    y = train['Category']
    y_num = label_to_num(y)
    test_id = test['Id']

    train_X = harmonize_data2(train)
    test_X = harmonize_data2(test)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        nthread=8,
        seed=229,
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
    )

    model_fit(xgb_model, train_X, y_num)
