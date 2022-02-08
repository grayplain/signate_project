import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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


def read_pd_data(file_name, header="infer"):
    return pd.read_table(os.getcwd() + '/datas/' + file_name, header=header, index_col=0)


def trim_data(pd_data):
    if type(pd_data) is not pd.core.frame.DataFrame:
        return
    dummy_pd_data = pd.get_dummies(pd_data)
    return dummy_pd_data


def _get_dummies(data):
    return pd.get_dummies(data, columns=['sex', 'pclass', 'embarked'])


def model_pred_pd_data(X, y, estimator):
    if not all([hasattr(estimator, 'fit'), hasattr(estimator, 'predict'), hasattr(estimator, 'score')]):
        print("estimator は学習器ではありません。")
        return

    # なるほど、パイプラインは一つの大きい estimator として扱うことができるのか。
    pipeline = pipe_line(estimator)

    print('estimator = {}'.format(estimator.__class__))
    # print('前処理なし = {}'.format(cross_validate(estimator, X, y)["test_score"]))
    # print('前処理あり = {}'.format(cross_validate(pipeline, X, y)["test_score"]))
    print('前処理なし = \n{}'.format(cross_validate(estimator, X, y)))
    print('前処理あり = \n{}'.format(cross_validate(pipeline, X, y)))

    print('----------')


def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])


def main():
    train_data = read_pd_data('train.tsv')
    test_data = read_pd_data('test.tsv')

    dummy_data = _get_dummies(train_data)
    X = dummy_data.drop('survived', axis=1)
    X = X.interpolate()
    y = dummy_data['survived']

    estimator = LogisticRegression(max_iter=1000)
    # estimator = RandomForestClassifier()
    # estimator = SVC()

    # MLPClassifier()は実行コストがかかりすぎる & 精度そんなによくないので除外
    estimators = [GaussianNB(),
                  LogisticRegression(max_iter=1000),
                  SVC(),
                  RandomForestClassifier(),
                  GradientBoostingClassifier()]

    for estimator in estimators:
         model_pred_pd_data(X.values, y.values, estimator)

    
main()
