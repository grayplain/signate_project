from PIL import Image
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier


def read_pd_data(file_name):
    file_name_path = '/' + file_name
    return pd.read_table(os.getcwd() + '/datas/' + file_name_path)


def open_image(file_name):
    file_name_path = 'train_images/' + file_name
    return np.array(Image.open(os.getcwd() + '/datas/' + file_name_path))


def load_image_datas(max_number):
    images = np.empty((0, 28*28))

    if max_number < 0:
        print('0未満の値の画像は存在しません。')
        return

    # debug 時はちょっと大きすぎる値をぶちこむとパソコンいかれるので制御。
    if max_number > 100:
        print('デバッグ時は、あんま重い処理は勘弁。')
        return

    for num in range(0, max_number):
        file_name = 'train_' + str(max_number) + '.jpg'
        img = open_image(file_name)
        # このままだと img は 28×28 次元の配列として扱われるので、reshape(-1, 784)を実行して
        # 784要素の1次元の配列に変換する。
        images = np.append(images, img.reshape(-1, 784), axis=0)
        # images = np.vstack((images, img.reshape(-1, 784)))

    return images


def count_number():
    num_pd = read_pd_data('train_master.tsv')
    print(num_pd[:50].groupby('category_id').count())



def train_image():
    data_count = 5
    num_pd = read_pd_data('train_master.tsv')
    y = num_pd[:data_count]['category_id'].values
    images = load_image_datas(data_count)
    X_train, X_test, y_train, y_test = train_test_split(images, y)
    print(X_train.shape)
    print(X_test.shape)
    print(y)

train_image()


def append_test():
    a = np.zeros((1, 5))
    # print(a)
    b = np.arange(5)
    print(b)
    print(b.reshape(-1, 5))
    print(b.reshape(-1, 5))
    # c = np.append(a, b.reshape(-1, 5), axis=0)
    # print(c)

    # a_2d = np.arange(6).reshape(2, 3)
    # print(a_2d)
    # a_2d_ex = np.arange(3).reshape(1, 3) * 10
    # print(a_2d_ex)
    # print(np.append(a_2d, a_2d_ex, axis=0))

# append_test()