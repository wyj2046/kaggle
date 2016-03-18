# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


dow = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}


def dates_to_seconds(dates):
    times = dates.split()[1]
    times_list = times.split(':')
    seconds = int(times_list[0]) * 3600 + int(times_list[1]) * 60 + int(times_list[2])
    return seconds


def harmonize_data(data):
    """
    modify source data
    """
    data['seconds'] = data['Dates'].apply(dates_to_seconds)
    data['hour'] = pd.to_datetime(data['Dates']).dt.hour
    data.drop('Dates', axis=1, inplace=True)

    if 'Category' in data:
        data.drop('Category', axis=1, inplace=True)
    if 'Descript' in data:
        data.drop('Descript', axis=1, inplace=True)
    if 'Resolution' in data:
        data.drop('Resolution', axis=1, inplace=True)

    if 'Id' in data:
        data.drop('Id', axis=1, inplace=True)

    # dummies_week = pd.get_dummies(data['DayOfWeek'], prefix='Week')
    # data = data.join(dummies_week)
    data['dow'] = data['DayOfWeek'].map(dow)
    data.drop('DayOfWeek', axis=1, inplace=True)

    dummies_pd = pd.get_dummies(data['PdDistrict'], prefix='Pd')
    data = data.join(dummies_pd)
    data.drop('PdDistrict', axis=1, inplace=True)

    data.drop('Address', axis=1, inplace=True)

    return data


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


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    y_train = train['Category']
    test_id = test['Id']

    X_train = harmonize_data2(train)
    X_test = harmonize_data2(test)

    # clf = RandomForestClassifier(n_jobs=1, n_estimators=100, min_samples_split=1000)
    # clf = GradientBoostingClassifier(n_estimators=100, min_samples_split=1000, verbose=1)
    # clf = SVC(probability=True, kernel='linear')
    # tunes_parameters = [
    #     {'kernel': ['rbf', 'linear']},
    #     {'C': [0.01, 0.1, 1, 10]}
    # ]

    # clf = SVC(probability=True)
    # clf = GridSearchCV(clf, tunes_parameters)
    # clf.fit(X_train, y_train)
    # print clf.best_params_
    # for params, mean_score, scores in clf.grid_scores_:
    #     print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std() * 2, params)

    # results = clf.predict_proba(X_test)

    # label_num = label_to_num(y_train)
    # xg_train = xgb.DMatrix(X_train.values, label=label_num.values)
    # xg_test = xgb.DMatrix(X_test.values)

    # param = {}
    # param['objective'] = 'multi:softprob'
    # param['eval_metric'] = 'mlogloss'
    # param['num_class'] = len(y_train.unique())
    # param['eta'] = 0.1
    # param['max_depth'] = 6
    # param['nthread'] = 2
    # param['silent'] = 1
    # param['seed'] = '229'

    # watchlist = [(xg_train, 'train')]
    # num_round = 1000
    # bst = xgb.train(param, xg_train, num_round, watchlist)
    # results = bst.predict(xg_test)

    xgb_model = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.1, n_estimators=2, max_depth=6, nthread=2, seed='229')
    xgb_model.fit(X_train, y_train, eval_metric='mlogloss')
    results = xgb_model.predict_proba(X_test)

    submission = pd.DataFrame(data=results, columns=y_train.unique())
    submission = submission.join(test_id)

    columns = ['Id'] + y_train.unique().tolist()
    submission.to_csv('result.csv', index=False, columns=columns)
