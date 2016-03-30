# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split


random_seed = 229
cv_folds = 3


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


def tune_xgb_param(X, y, xgbcv=False, sklearn_cv=False):
    base_param = {}
    base_param['nthread'] = 4
    base_param['silent'] = 1
    base_param['seed'] = random_seed
    base_param['objective'] = 'multi:softprob'

    base_param['learning_rate'] = 0.1
    base_param['n_estimators'] = 985
    base_param['max_depth'] = 5
    base_param['min_child_weight'] = 7
    base_param['gamma'] = 0.2
    base_param['subsample'] = 0.8
    base_param['colsample_bytree'] = 0.8

    if xgbcv:
        xg_train = xgb.DMatrix(X, label=y)
        cv_result = xgb.cv(base_param, xg_train, num_boost_round=base_param['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=50, verbose_eval=1, show_stdv=False, seed=random_seed, stratified=True)
        base_param['n_estimators'] = cv_result.shape[0]

    tune_param = {}
    # tune_param['max_depth'] = [6, 7, 8, 9]
    # tune_param['min_child_weight'] = range(5, 10, 1)
    # tune_param['min_child_weight'] = range(9, 20, 2)

    # tune_param['gamma'] = [i / 10.0 for i in range(0, 3)]
    # tune_param['gamma'] = [i / 10.0 for i in range(3, 6)]
    # tune_param['gamma'] = [i / 100.0 for i in range(15, 25, 2)]
    # tune_param['gamma'] = [i / 100.0 for i in range(23, 35, 2)]

    tune_param['subsample'] = [i / 10.0 for i in range(5, 10)]
    # tune_param['colsample_bytree'] = [i / 10.0 for i in range(6, 10)]
    # tune_param['colsample_bytree'] = [i / 100.0 for i in range(75, 95)]

    # tune_param['learning_rate'] = [0.03, 0.04, 0.05]
    # tune_param['n_estimators'] = [200 + i * 10 for i in range(0, 11)]

    model = xgb.XGBClassifier(**base_param)

    if sklearn_cv:
        clf = GridSearchCV(model, tune_param, scoring='log_loss', n_jobs=2, cv=cv_folds, verbose=2)
        clf.fit(X, y)
        for item in clf.grid_scores_:
            print item
        print 'BEST', clf.best_params_, clf.best_score_

        return clf.best_estimator_

    return model


def get_pred_y1(train_X, train_y, test_X):
    X_fit, X_eval, y_fit, y_eval = train_test_split(train_X, train_y, test_size=0.1, random_state=random_seed)

    xgb_model = tune_xgb_param(train_X, train_y, False, True)

    # xgb_model.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric='mlogloss', eval_set=[(X_fit, y_fit), (X_eval, y_eval)])
    xgb_model.fit(train_X, train_y, early_stopping_rounds=50, eval_metric='mlogloss', eval_set=[(train_X, train_y)])

    pred_y = xgb_model.predict_proba(test_X)
    return pred_y


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train_y = train['Category']
    train_y = label_to_num(train_y)
    test_id = test['Id']

    train_X = harmonize_data2(train)
    test_X = harmonize_data2(test)

    # clf = RandomForestClassifier(n_jobs=1, n_estimators=100, min_samples_split=1000)
    # clf = GradientBoostingClassifier(n_estimators=100, min_samples_split=1000, verbose=1)
    # clf = SVC(probability=True, kernel='linear')
    # tunes_parameters = [
    #     {'kernel': ['rbf', 'linear']},
    #     {'C': [0.01, 0.1, 1, 10]}
    # ]

    # clf = SVC(probability=True)
    # clf = GridSearchCV(clf, tunes_parameters)
    # clf.fit(train_X, train_y)
    # print clf.best_params_
    # for params, mean_score, scores in clf.grid_scores_:
    #     print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std() * 2, params)

    # results = clf.predict_proba(test_X)

    results = get_pred_y1(train_X, train_y, test_X)

    submission = pd.DataFrame(data=results, columns=train['Category'].unique())
    submission = submission.join(test_id)

    columns = ['Id'] + train['Category'].unique().tolist()
    submission.to_csv('result.csv', index=False, columns=columns)
