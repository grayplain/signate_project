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

    print('前処理なし = {}'.format(cross_validate(estimator, X, y)["test_score"].mean()))
    print('前処理あり = {}'.format(cross_validate(pipeline, X, y)["test_score"].mean()))
    # print('前処理なし = \n{}'.format(cross_validate(estimator, X, y)))
    # print('前処理あり = \n{}'.format(cross_validate(pipeline, X, y)))

    print('----------')


def pipe_line(estimator):
    return Pipeline([('StandardScaler', StandardScaler()),
                     ('Estimator', estimator)
                     ])


def choose_candiate():
    train_data = read_pd_data('train.tsv')

    dummy_data = _get_dummies(train_data)
    X = dummy_data.drop('survived', axis=1)
    X = X.interpolate()
    y = dummy_data['survived']

    # MLPClassifier()は実行コストがかかりすぎる & 精度そんなによくないので除外
    estimators = [GaussianNB(),
                  LogisticRegression(max_iter=1000),
                  SVC(),
                  RandomForestClassifier(),
                  GradientBoostingClassifier()]

    for estimator in estimators:
         model_pred_pd_data(X.values, y.values, estimator)

def file_path(file_name):
    return 'titanic' + '/' + file_name


def make_output():
    train_data = read_pd_data(file_path('train.tsv'))
    print(train_data)
    return
    dummy_data = _get_dummies(train_data)
    X = dummy_data.drop('survived', axis=1)
    X = X.interpolate()
    y = dummy_data['survived']

    test_data = read_pd_data(file_path('test.tsv'))
    dummy_test_data = _get_dummies(test_data)
    X_test = dummy_test_data.interpolate()

    estimator = SVC(probability=True)

    pipeline = pipe_line(estimator).fit(X.values, y.values)
    pred = pipeline.predict_proba(X_test)

    sample_submit = read_pd_data(file_path('sample_submit.tsv'), header=None)
    pred = pipeline.predict_proba(X_test)[:, 1]
    sample_submit[1] = pred
    sample_submit[1] = sample_submit[1].round(2)
    print(sample_submit[1].round(2))
    sample_submit.to_csv('submit2.tsv', header=None, sep='\t')



# make_output()
standard_scaler_test()