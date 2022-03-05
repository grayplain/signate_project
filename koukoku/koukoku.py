import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

def model_pred_pd_data(X, y, estimator):
    if not all([hasattr(estimator, 'fit'), hasattr(estimator, 'predict'), hasattr(estimator, 'score')]):
        print("estimator は学習器ではありません。")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    logloss = log_loss(estimator.fit(X_train, y_train).predict(X_test), y_test, labels=[0, 1])

    print('estimator = {}'.format(estimator.__class__))
    print('結果(推定器単体) =    {:.2f}'.format(logloss))
    # print('前処理なし = {}'.format(cross_validate(estimator, X, y)["test_score"].mean()))
    print('----------')

def choose_candiate(X, y):
    # MLPClassifier()は実行コストがかかりすぎる & 精度そんなによくないので除外
    estimators = [GaussianNB(),
                  LogisticRegression(max_iter=1000)]

    for estimator in estimators:
         model_pred_pd_data(X.values, y.values, estimator)

def main():

    # train_data = read_pd_data("train_sample.tsv")
    train_data = read_pd_data("train.tsv")

    # これはターゲットなので、必ず drop。
    X = train_data.drop('click', axis=1)

    # とりあえず他の要素と相関係数が 0.7以上ある項目は drop してみる。
    X = train_data.drop(columns=['C1', 'I7'], axis=1)

    dummy_data = to_dummies(train_data, ['I1'])

    # 欠損値どうしようか。
    X = X.fillna(0)
    # print(X)

    y = dummy_data['click']

    choose_candiate(X, y)


main()


