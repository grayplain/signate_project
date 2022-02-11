import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import GaussianNB

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

    # なるほど、パイプラインは一つの大きい estimator として扱うことができるのか。
    pipeline = pipe_line(estimator)

    print('estimator = {}'.format(estimator.__class__))

    print('前処理なし = {}'.format(cross_validate(estimator, X, y)["test_score"].mean()))
    print('前処理あり = {}'.format(cross_validate(pipeline, X, y)["test_score"].mean()))
    # print('前処理なし = \n{}'.format(cross_validate(estimator, X, y)))
    # print('前処理あり = \n{}'.format(cross_validate(pipeline, X, y)))

    print('----------')


def file_path(file_name):
    return 'bentou' + '/' + file_name

def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])


# ワンホットエンコーディング変数に向いてない特徴量を落としたデータ。
# 超重要な可能性が高い弁当名の情報を丸っと落としているので、最終候補にはデータとして使えないす。
def trim_train_data_drop_bentou_names():
    train_data = read_pd_data(file_path('train.csv'))
    train_data = train_data.drop('name', axis=1)
    train_data = train_data.drop('remarks', axis=1)
    train_data = train_data.drop('event', axis=1)

    dummy_data = _get_dummies(train_data)
    X = dummy_data

    X = X.fillna(0)
    X = X.replace({'precipitation': {"--": "0"}})
    # print(X)
    print(X.corr())
    # X.to_csv('trim_train.tsv', sep='\t')
    X.corr().to_csv('corr_train.tsv', sep='\t')


# 弁当名にキーワードを追加してワンホットエンコーディングしたやつ。
def trim_train_data():
    train_data = read_pd_data(file_path('train.csv'))
    train_data = train_data.drop('name', axis=1)
    train_data = train_data.drop('remarks', axis=1)
    train_data = train_data.drop('event', axis=1)

    one_hot_bentou = generate_one_hot_bentou_name()

    train_data_with_bentou = pd.concat([train_data, one_hot_bentou], axis=1, join='inner')
    dummy_data = _get_dummies(train_data_with_bentou)
    X = dummy_data

    X = X.fillna(0)
    X = X.replace({'precipitation': {"--": "0"}})
    # print(X)
    print(X.corr())
    # X.to_csv('trim_train.tsv', sep='\t')
    X.corr().to_csv('corr_train.tsv', sep='\t')


def generate_one_hot_bentou_name():
    train_data = pd.read_csv(os.getcwd() + '/datas/' + file_path('train.csv'),
                             usecols=['datetime', 'y', 'name'])
    keywords = ["白身魚", "カレー", "カツ", "かつ", "チキン", "鶏",  "ハンバーグ", "カレイ"]
    countArrays = array_keyword_if_contain(train_data['name'].values, keywords)

    return pd.DataFrame(countArrays.T, columns=keywords, index=train_data['datetime'])


def array_keyword_if_contain(bentouArray, keywords):
    countArrays = []
    for keyword in keywords:
        countArray = []
        for name in bentouArray:
            if keyword in name:
                countArray.append(1)
            else:
                countArray.append(0)
        countArrays.append(countArray)

    retValue = np.array(countArrays)
    return retValue



trim_train_data()