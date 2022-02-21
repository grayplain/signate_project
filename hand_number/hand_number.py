from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


# 提出ファイル作成
def output_submit(test_data, estimator):
    sample_submit = read_pd_data('sample_submit.tsv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_number.tsv', header=None, sep='\t')

def read_pd_data(file_name, header="infer"):
    file_name_path = '/' + file_name
    return pd.read_table(os.getcwd() + '/datas/' + file_name_path, header=header)

def write_numpy_data(file_name, file):
    file_name_path = '/' + file_name
    np.savetxt(os.getcwd() + '/datas/' + file_name_path, file, fmt='%.1f')

def write_fitted_model(model, file_name='fitted_model.pickle'):
    with open(file_name, mode='wb') as f:
        pickle.dump(model, f, protocol=3)

def load_fitted_model(file_name='fitted_model.pickle'):
    return pickle.load(open(file_name, 'rb'))

def open_image(file_name, is_train_data):
    env = 'train_' if is_train_data else 'test_'

    file_name_path = env + 'images/' + file_name
    return np.array(Image.open(os.getcwd() + '/datas/' + file_name_path))


def load_image_datas(max_number, is_train_data=True):
    env = 'train_' if is_train_data else 'test_'

    images = np.empty((0, 28*28))

    if max_number < 0:
        print('0未満の値の画像は存在しません。')
        return

    for num in range(max_number):
        file_name = env + str(num) + '.jpg'
        img = open_image(file_name, is_train_data)
        # このままだと img は 28×28 次元の配列として扱われるので、reshape(-1, 784)を実行して
        # 784要素の1次元の配列に変換する。
        images = np.append(images, img.reshape(-1, 784), axis=0)
        # images = np.vstack((images, img.reshape(-1, 784)))

    return images


def make_image_pipe_line(estimator):
    return Pipeline(steps=[('Scaler', StandardScaler()),
                           ('Estimator', estimator)])

def classifier_number_from_images(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    # estimators = [RandomForestClassifier(), LogisticRegression(max_iter=100000), SVC(max_iter=100000)]
    estimators = [MLPClassifier()]
    # estimators = [SVC()]

    print('全データ数 = {}'.format(X.shape[0]))
    for estimator in estimators:
        print('推定器 =           {}'.format(estimator.__class__))
        pipe = make_image_pipe_line(estimator)
        print('パイプ =       {:.2f}'.format(pipe.fit(X_train, y_train).score(X_test, y_test)))
        print('--------')


def fit_image_model(X, y, estimator):
    pipe = make_image_pipe_line(estimator)
    return pipe.fit(X, y)


def estimate_image(data_count):
    num_pd = read_pd_data('train_master.tsv')
    y = num_pd[:data_count]['category_id'].values
    images = load_image_datas(data_count)
    classifier_number_from_images(images, y)

def main():
    #学習済みモデルの読み込み、予測
    test_data_count = 10000
    test_images = load_image_datas(test_data_count, is_train_data=False)
    load_model = load_fitted_model()
    output_submit(test_images, load_model)


def train_production_model():
    # 学習済みモデルの作成、保存
    train_data_count = 59999
    train_num_pd = read_pd_data('train_master.tsv')
    y = train_num_pd[:train_data_count]['category_id'].values
    train_images = load_image_datas(train_data_count)
    fitted_model = fit_image_model(train_images, y, MLPClassifier())
    write_fitted_model(fitted_model)



main()