import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def output_submit(test_data, estimator):
    sample_submit = read_pd_data('submission.csv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_phone.csv', header=None, sep=',', index=False)

def read_pd_data(file_name, header="infer"):
    file_name_path = '/' + file_name
    return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header)

def display_init():
    pd.set_option('display.max_columns', 100)
    pd.options.display.precision = 4


def trim_pddata(pd_data):
    trim_data = pd_data.drop('id', axis=1)
    return trim_data


def inspect_pd_data(pd_data):
    # print(pd_data.corr())
     print(pd_data.min())
     print(pd_data.max())
     print(pd_data.mean())
     print(pd_data.median())


def fit_model(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
#     特徴量エンジニアリングかます
    estimator = RandomForestClassifier()

    estimator.fit(X_train, y_train)
    phone_f1_score = f1_score(y_test, estimator.predict(X_test), average='macro')
    print(phone_f1_score)

    return estimator


def main():
    display_init()
    phone_train_pd_data = trim_pddata(read_pd_data("train.csv"))
    # inspect_pd_data(phone_train_pd_data)
    fitted_model = fit_model(phone_train_pd_data.drop('price_range', axis=1), phone_train_pd_data['price_range'])

    output_submit(trim_pddata(read_pd_data("test.csv")), fitted_model)



main()






