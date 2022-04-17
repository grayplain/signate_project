import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def add_feature_pddata(pd_data: pd.DataFrame):
    # pd_data['isGarake'] = pd_data['px_height'].apply(lambda x : 1 if x < 100 else 0)
    return pd_data



def inspect_pd_data(pd_data):
    print(pd_data.corr())
     # print(pd_data.min())
     # print(pd_data.max())
     # print(pd_data.mean())
     # print(pd_data.median())


def make_pipe_line(estimator):
    return Pipeline(steps=[('Scaler', StandardScaler()),
                           ('Estimator', estimator)])


def fit_model(X_data, y_data, estimator):
    estimator = make_pipe_line(estimator)
    estimator.fit(X_data, y_data)

    return estimator


def test_model(fitted_model):
    phone_test_pd_data = trim_pddata(read_pd_data("test.csv"))
    phone_test_pd_data = add_feature_pddata(phone_test_pd_data)
    output_submit(phone_test_pd_data, fitted_model)


def main():
    display_init()
    phone_train_pd_data = trim_pddata(read_pd_data("train.csv"))
    # huga = phone_train_pd_data.query('price_range == 3')
    # inspect_pd_data(phone_train_pd_data)
    # return

    phone_train_pd_data = add_feature_pddata(phone_train_pd_data)
    train_X_data = phone_train_pd_data.drop('price_range', axis=1)
    train_y_data = phone_train_pd_data['price_range']
    estimator = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1)
    # estimator = RidgeClassifier()

    num_of_fit = 10
    macro_f1_ave = 0
    macro_f1_max = 0
    max_fitted_model = estimator
    for fit_count in range(0, num_of_fit):
        X_train, X_test, y_train, y_test = train_test_split(train_X_data, train_y_data)
        fitted_model = fit_model(X_train, y_train, estimator)
        phone_f1_score = f1_score(y_test, fitted_model.predict(X_test), average='macro')

        if phone_f1_score > macro_f1_max:
            macro_f1_max = phone_f1_score
            max_fitted_model = fitted_model

        print('macroF1_score = {:.4f}'.format(phone_f1_score))
        macro_f1_ave += phone_f1_score
    print('macroF1_average = {:.4f}'.format(macro_f1_ave / num_of_fit))
    print('macroF1_max = {:.4f}'.format(macro_f1_max))

    test_model(max_fitted_model)



main()
