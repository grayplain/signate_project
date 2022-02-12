import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import bentou_feature_engineering


import os

# 標準化の使い方がよくわからんかったので、試しに書いた。
from sklearn.datasets import make_blobs


def standard_scaler_test():
    X, y = make_blobs()
    scaled_X = StandardScaler().fit_transform(X)
    print(scaled_X)


# 提出ファイル作成
def output_submit(test_data, estimator):
    pipeline = pipe_line(estimator)

    sample_submit = read_pd_data('sample_submit.tsv', header=None)
    pred = pipeline.predict_proba(test_data)[:, 1]
    sample_submit[1] = pred
    sample_submit.to_csv('submit.tsv', header=None, sep='\t')


def read_pd_data(file_name):
    return pd.read_csv(os.getcwd() + '/datas/' + file_name, index_col=0)


def trim_data(pd_data):
    if type(pd_data) is not pd.core.frame.DataFrame:
        return
    dummy_pd_data = pd.get_dummies(pd_data)
    return dummy_pd_data


def _get_dummies(data):
    return pd.get_dummies(data, columns=['weather', 'week'])


def model_pred_pd_data(X, y, estimator):
    if not all([hasattr(estimator, 'fit'), hasattr(estimator, 'predict'), hasattr(estimator, 'score')]):
        print("estimator は学習器ではありません。")
        return

    pipeline = pipe_line(estimator)

    pred_estimator_value = estimator.fit(X, y).predict(X)
    pred_pipe_value = pipeline.fit(X, y).predict(X)
    true_value = y.values

    # TODO: 交差検証などしないと、過学習してしまっている。
    print('estimator = {}'.format(estimator.__class__))
    print('前処理なし = {}'.format(np.sqrt(mean_squared_error(true_value, pred_estimator_value))))
    print('前処理あり = {}'.format(np.sqrt(mean_squared_error(true_value, pred_pipe_value))))
    print('----------')


def file_path(file_name):
    return 'bentou' + '/' + file_name

def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])



# 弁当名にキーワードを追加してワンホットエンコーディングしたやつ。
def trim_train_data():
    train_data = read_pd_data(file_path('train.csv'))
    one_hot_fe = bentou_feature_engineering.Bentou_Name_OneHot()

    one_hot_bentou_df = one_hot_fe.generate_one_hot_bentou_name(pd.DataFrame(train_data['name']))

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

    model_pred_pd_data(X, y, RandomForestRegressor())


trim_train_data()