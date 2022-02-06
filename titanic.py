import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os


def main():
    train_data = read_pd_data('train.tsv')
    test_data = read_pd_data('test.tsv')

    mini_test_data = test_data[['sibsp', 'parch', 'fare']]
    mini_train_data = train_data[['survived', 'sibsp', 'parch', 'fare']]
    y = mini_train_data['survived']
    X = mini_train_data.drop(['survived'], axis=1)
    logi = LogisticRegression().fit(X, y)
    output_submit(logi, mini_test_data)


def output_submit(model, test_data):
    sample_submit = read_pd_data('sample_submit.tsv', header=None)
    pred = model.predict_proba(test_data)[:, 1]
    sample_submit[1] = pred
    sample_submit.to_csv('submit.tsv', header=None, sep='\t')

def read_pd_data(file_name, header="infer"):
    return pd.read_table(os.getcwd() + '/datas/' + file_name, header=header, index_col=0)


def trim_data(pd_data):
    if type(pd_data) is not pd.core.frame.DataFrame:
        return
    dummy_pd_data = pd.get_dummies(pd_data)
    return dummy_pd_data


main()
