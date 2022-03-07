import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import os


def file_path(file_name):
    return 'koukoku' + '/' + file_name


# 提出ファイル作成
def output_submit(result, file_name='submit_blank.csv'):
    sample_submit = read_pd_data(file_path('sample_submit.csv'), headerFlag=None)
    sample_submit[1] = result
    sample_submit.to_csv(file_name, header=None, sep=',')


def read_pd_data(file_name, headerFlag="infer"):
    return pd.read_table(os.getcwd() + '/datas/' + file_name, index_col=0, header=headerFlag)


def to_dummies(data, column_name):
    return pd.get_dummies(data=data, columns=column_name)


def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])


def model_pred_pd_data(X, y, estimator):
    if not all([hasattr(estimator, 'fit'), hasattr(estimator, 'predict'), hasattr(estimator, 'score')]):
        print("estimator は学習器ではありません。")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

    logloss_estimate = log_loss(y_test, estimator.fit(X_train, y_train).predict_proba(X_test))
    logloss_pipe = log_loss(y_test, pipe_line(estimator).fit(X_train, y_train).predict_proba(X_test))

    print('estimator = {}'.format(estimator.__class__))
    # print('結果(推定器単体) =    {:.2f}'.format(logloss_estimate))
    print('結果(パイプ) =    {:.2f}'.format(logloss_pipe))
    # print('前処理なし = {}'.format(cross_validate(estimator, X, y)["test_score"].mean()))
    print('----------')


def choose_candiate(X, y):
    # MLPClassifier()は実行コストがかかりすぎる & 精度そんなによくないので除外
    # estimators = [GaussianNB(),
    #               LogisticRegression(max_iter=1000)]
    estimators = [LogisticRegression(max_iter=1000)]

    for estimator in estimators:
        model_pred_pd_data(X.values, y.values, estimator)


def main():
    pd.set_option('display.max_columns', 100)
    pd.options.display.precision = 2
    pd.options.display.float_format = '{:.2f}'.format

    train_data = read_pd_data("train.tsv")
    # train_data = read_pd_data("train_sample.tsv")
    # 欠損値どうしようか。
    train_data = train_data.fillna(0)

    # これはターゲットなので、必ず drop。
    X = train_data.drop('click', axis=1)

    # とりあえず他の要素と相関係数が 0.7以上ある項目は drop してみる。

    # dummy_data = to_dummies(train_data, ['I1', 'I3', 'I4', 'I6', 'I7', 'I8', 'I9', 'I10'])

    y = train_data['click']

    # 推定器をいい感じに調べる
    # choose_candiate(X, y)


def predict_click():
    pd.set_option('display.max_columns', 100)
    pd.options.display.precision = 2
    pd.options.display.float_format = '{:.2f}'.format

    train_data = read_pd_data("train.tsv")
    # train_data = read_pd_data("train_sample.tsv")
    # 欠損値どうしようか。
    train_data = train_data.fillna(0)

    # これはターゲットなので、必ず drop。
    X = train_data.drop('click', axis=1)

    # とりあえず他の要素と相関係数が 0.7以上ある項目は drop してみる。

    # dummy_data = to_dummies(train_data, ['I1', 'I3', 'I4', 'I6', 'I7', 'I8', 'I9', 'I10'])

    y = train_data['click']

    # 推定器をいい感じに調べる
    # choose_candiate(X, y)

    # 出力
    # test_data = read_pd_data("test.tsv")
    # X_test = test_data.fillna(0)
    #
    # sample_submit = pd.read_csv(os.getcwd() + '/datas/' + 'sample_submit.csv', header=None, index_col=0)
    # logloss_pipe = pipe_line(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(X_test)
    #
    # sample_submit[1] = logloss_pipe[:, 1]
    # sample_submit[1] = sample_submit[1].round(4)
    # print(sample_submit[1].round(4))
    # sample_submit.to_csv('submit2.tsv', header=None, sep=',')

predict_click()