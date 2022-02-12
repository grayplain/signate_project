import numpy as np
from numpy.random import *
import pandas as pd
import datetime
import math

import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

import bentou_feature_engineering


import os

# 標準化の使い方がよくわからんかったので、試しに書いた。
from sklearn.datasets import make_blobs


def standard_scaler_test():
    X, y = make_blobs()
    scaled_X = StandardScaler().fit_transform(X)
    print(scaled_X)


# 提出ファイル作成
def output_submit(result, file_name='submit_blank.csv'):
    sample_submit = read_pd_data(file_path('sample_submit.csv'), headerFlag=None)
    sample_submit[1] = result
    sample_submit.to_csv(file_name, header=None, sep=',')


def read_pd_data(file_name, headerFlag="infer"):
    return pd.read_csv(os.getcwd() + '/datas/' + file_name, index_col=0, header=headerFlag)


def trim_data(pd_data):
    if type(pd_data) is not pd.core.frame.DataFrame:
        return
    dummy_pd_data = pd.get_dummies(pd_data)
    return dummy_pd_data


def _get_dummies(data):
    return pd.get_dummies(data, columns=['weather', 'week'])

def rmse_score(y_true, y_pred):
    """RMSE (Root Mean Square Error: 平均二乗誤差平方根) を計算する関数"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return rmse


def model_pred_pd_data(X, y, estimator):
    if not all([hasattr(estimator, 'fit'), hasattr(estimator, 'predict'), hasattr(estimator, 'score')]):
        print("estimator は学習器ではありません。")
        return

    score_funcs = {
        'rmse': make_scorer(rmse_score),
    }

    pipeline = pipe_line(estimator)

    pred_estimator_value = cross_validate(estimator, X, y, cv=KFold(n_splits=5), scoring=score_funcs)
    pred_pipeline_value = cross_validate(pipeline, X, y, cv=KFold(n_splits=5), scoring=score_funcs)

    print('estimator = {}'.format(estimator.__class__))
    print('前処理なし = \n{}'.format(pred_estimator_value['test_rmse']))
    print('前処理あり = \n{}'.format(pred_pipeline_value['test_rmse']))
    print('----------')


def file_path(file_name):
    return 'bentou' + '/' + file_name

def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])

def trim_bentou_data(one_hot_bentou_df, train_data):
    train_data = train_data.drop('name', axis=1)
    train_data = train_data.drop('remarks', axis=1)
    train_data = train_data.drop('event', axis=1)
    train_data_with_bentou = pd.concat([train_data, one_hot_bentou_df], axis=1, join='inner')
    dummy_data = _get_dummies(train_data_with_bentou)
    X = dummy_data
    X = X.fillna(0)
    X = X.replace({'precipitation': {"--": "0"}})
    y = X['y']
    X = X.drop('y', axis=1)
    # TODO: テストデータに雪と雷がないので訓練データから情報を落としているが、
    #  このやり方だとせっかく手に入れた情報量を落としてもったいなさそうなので、何か検討する。
    X = X.drop('weather_雪', axis=1)
    X = X.drop('weather_雷電', axis=1)
    return X, y


# 弁当名にキーワードを追加してワンホットエンコーディングしたやつ。
def start_ml():
    train_data = read_pd_data(file_path('train.csv'))
    one_hot_fe = bentou_feature_engineering.Bentou_Name_OneHot()

    one_hot_bentou_df = one_hot_fe.generate_one_hot_bentou_name(pd.DataFrame(train_data['name']))
    X, y = trim_bentou_data(one_hot_bentou_df, train_data)

    # estimators = [SVR(), Ridge(alpha=1.0), RandomForestRegressor()]
    # estimators = [SVR(C=100.0)]
    # for estimator in estimators:
    #     model_pred_pd_data(X, y, estimator)

    trial_model(one_hot_fe, X, y, Ridge(alpha=1.0))



def trial_model(one_hot_bentou_name, X_train, y_train, estimator):
    test_data = read_pd_data(file_path('test.csv'))
    one_hot_bentou_test_df = one_hot_bentou_name.generate_one_hot_bentou_name(pd.DataFrame(test_data['name']))
    test_data = test_data.drop('name', axis=1)
    test_data = test_data.drop('remarks', axis=1)
    test_data = test_data.drop('event', axis=1)
    test_data_with_bentou = pd.concat([test_data, one_hot_bentou_test_df], axis=1, join='inner')
    dummy_test_data = _get_dummies(test_data_with_bentou)
    X_test = dummy_test_data
    X_test = X_test.fillna(0)
    X_test = X_test.replace({'precipitation': {"--": "0"}})
    pred_value = estimator.fit(X_train, y_train).predict(X_test)
    dt_now = datetime.datetime.now()
    output_submit(pred_value, 'result' + dt_now.strftime('%Y%m%d_%H%M') + '.csv')


start_ml()
