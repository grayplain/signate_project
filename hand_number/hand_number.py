from PIL import Image
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from skimage import filters
from sklearn.datasets import load_digits


def read_pd_data(file_name):
    file_name_path = '/' + file_name
    return pd.read_table(os.getcwd() + '/datas/' + file_name_path)

def write_numpy_data(file_name, file):
    file_name_path = '/' + file_name
    np.savetxt(os.getcwd() + '/datas/' + file_name_path, file, fmt='%.1f')


def open_image(file_name):
    file_name_path = 'train_images/' + file_name
    return np.array(Image.open(os.getcwd() + '/datas/' + file_name_path))


def load_image_datas(max_number):
    images = np.empty((0, 28*28))

    if max_number < 0:
        print('0未満の値の画像は存在しません。')
        return

    # debug 時はちょっと大きすぎる値をぶちこむとパソコンいかれるので制御。
    # if max_number > 200:
    #     print('デバッグ時は、あんま重い処理は勘弁。')
    #     return

    for num in range(1, max_number + 1):
        file_name = 'train_' + str(num) + '.jpg'
        img = open_image(file_name)
        # このままだと img は 28×28 次元の配列として扱われるので、reshape(-1, 784)を実行して
        # 784要素の1次元の配列に変換する。
        images = np.append(images, img.reshape(-1, 784), axis=0)
        # images = np.vstack((images, img.reshape(-1, 784)))

    return images


def count_number(num):
    num_pd = read_pd_data('train_master.tsv')
    print(num_pd[:num].groupby('category_id').count())

# 弁当屋やタイタニックのパイプラインを使えるが、勉強のためもう一回1から作る
def make_image_pipe_line(estimator):
    # return Pipeline(steps=[('PCA', PCA()),
    #                        ('Scaller', StandardScaler()),
    #                        ('Estimator', estimator)])
    return Pipeline(steps=[('Scaller', StandardScaler()),
                           ('Estimator', estimator)])
    # return Pipeline(steps=[('NMF', NMF(n_components=15, random_state=0)),
    #                        ('Scaller', StandardScaler()),
    #                        ('Estimator', estimator)])


def classifier_number_from_images(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    estimators = [RandomForestClassifier(), LogisticRegression(), SVC()]

    for estimator in estimators:
        pipe = make_image_pipe_line(estimator)
        print(pipe.fit(X_train, y_train).score(X_train, y_train))


def main():
    print("")
    data_count = 500
    num_pd = read_pd_data('train_master.tsv')
    y = num_pd[:data_count]['category_id'].values
    images = load_image_datas(data_count)
    classifier_number_from_images(images, y)

    # サンプルはよく認識するのー。
    # digits = load_digits()
    # classifier_number_from_images(digits.data, digits.target)

main()



def digit_test():

    np.set_printoptions(threshold=1000)

    digits = load_digits()
    print('digit_data_shape = {}'.format(digits.data.shape))
    print('digits.target.shape = {}'.format(digits.target.shape))
    print('digit_data[0] = {}'.format(digits.data[0]))
    image = Image.fromarray(digits.data[1].reshape(8, 8))
    Image._show(image)

# digit_test()


def wakarima():
    # data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    # data = [[0, 50],[5, 40],[3, 30]]
    data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    print(scaler.fit_transform(data))
    return

    data_count = 1
    images = load_image_datas(data_count)
    digits = load_digits()

    images_min_max = MinMaxScaler().fit_transform(images)
    # digit_min_max = MinMaxScaler().fit_transform(digits.data)
    print(images_min_max)
    # print(digit_min_max)


    print("hoohoohohohoh")
    # write_numpy_data("simasa.txt", MinMaxScaler().fit_transform(images)[0])
    # write_numpy_data("simasa2.txt", MinMaxScaler().fit_transform(digits.data)[0])

# wakarima()