from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
import time
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d as patches
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from sklearn.decomposition import NMF


# 提出ファイル作成
def output_submit(test_data, estimator):
    sample_submit = read_pd_data('sample_submit.csv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_minting.csv', header=None, sep=',', index=False)

def read_pd_data(file_name, header="infer"):
    file_name_path = '/' + file_name
    return pd.read_csv(os.getcwd() + '/datas/' + file_name_path, header=header)

def write_numpy_data(file_name, file):
    file_name_path = '/' + file_name
    np.savetxt(os.getcwd() + '/datas/' + file_name_path, file, fmt='%.1f')

def write_fitted_model(model, file_name='fitted_model.pickle'):
    with open(file_name, mode='wb') as f:
        pickle.dump(model, f, protocol=3)

def load_fitted_model(file_name='fitted_model.pickle'):
    return pickle.load(open(file_name, 'rb'))

def make_image_pipe_line(estimator):
    return Pipeline(steps=[('Scaler', StandardScaler()),
                           ('Estimator', estimator)])

def write_fitted_model(model, file_name='fitted_model.pickle'):
    with open(file_name, mode='wb') as f:
        pickle.dump(model, f, protocol=3)


def open_image(file_name, is_train_data):
    env = 'train_' if is_train_data else 'test_'

    file_name_path = env + 'data/' + file_name
    return np.array(Image.open(os.getcwd() + '/datas/' + file_name_path))


def load_patched_image_data(max_number=10, is_train_data=True):
    images = load_image_datas(max_number, is_train_data=is_train_data)
    patched_images = []

    for i in range(max_number):
        pre_patch = images[i].reshape(-1, 300)
        patched_image = patches(pre_patch, (30, 30))
        hoge = patched_image.reshape(-1, 1).shape
        huga2 = patched_image.reshape(-1, 1).transpose().shape
        print(huga2)
        continue
        patched_images.append(patched_image.reshape(-1, 1).shape)
    return np.asarray(patched_images)
    # 細分化した画像の復元確認用。
    # reconstructed = reconstruct_from_patches_2d(patched_images, pre_patch.shape)
    # print(patched_images.shape)
    # print(reconstructed.shape)
    # Image._show(Image.fromarray(reconstructed))


def load_image_datas(max_number, is_train_data=True):
    pixel = 300*300
    env = 'train_data_' if is_train_data else 'test_data_'

    images = np.empty((0, pixel))
    images_list = images.tolist()

    if max_number < 0:
        print('0未満の値の画像は存在しません。')
        return

    for num in range(max_number):
        file_name = env + str(num + 1) + '.jpg'
        img = open_image(file_name, is_train_data)
        # グレースケール化
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # canny = cv2.Canny(gray_scale, 100, 200)
        images_list.append(gray_scale)


    images = np.asarray(images_list)
    return images


def classifier_number_from_images(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    # estimators = [MLPClassifier(max_iter=5000)]
    estimators = [GaussianNB(), SVC(), RidgeClassifier(), RandomForestClassifier()]

    # return
    print('全データ数 = {}'.format(X.shape[0]))
    for estimator in estimators:
        start_time = time.time()
        print('推定器 =           {}'.format(estimator.__class__))
        pipe = make_image_pipe_line(estimator)
        print('結果(推定器単体) =    {:.2f}'.format(estimator.fit(X_train, y_train).score(X_test, y_test)))
        print('結果(パイプ) =       {:.2f}'.format(pipe.fit(X_train, y_train).score(X_test, y_test)))
        print("学習時間 = {:.2f}秒".format(time.time() - start_time))
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
    train_data_count = 5

    # 訓練用
    # 画像データ読み込み

    raw_images = load_image_datas(train_data_count)
    print('raw_images.shape= {}'.format(raw_images.shape))
    print('raw_images[0].shape= {}'.format(raw_images[0].shape))

    train_image_datas = load_patched_image_data(max_number=5)
    print('train_image_datas.shape= {}'.format(train_image_datas.shape))
    print('train_image_datas[0].shape= {}'.format(train_image_datas[0].shape))
    print('train_image_datas[0].reshape(-1,1).shape= {}'.format(train_image_datas[0].reshape(-1, 1).shape))
    return

    # 正解データの読み込み
    num_pd = read_pd_data('train_aug.csv')
    y = num_pd[:train_data_count]['target'].values

    nmf = NMF(n_components=30, init='random', random_state=0)
    nmf.fit(train_image_datas)

    return

    # # 予測してどの推定器が一番良さげか。
    classifier_number_from_images(reshaped_data, y)

    # 学習済みモデルの作成
    # fitted_model = fit_image_model(reshaped_data, y, MLPClassifier())
    # write_fitted_model(fitted_model)


    # # 本番用
    # test_data_count = 100
    # test_image_datas = load_image_datas(test_data_count, is_train_data=False)
    # test_reshaped_data = test_image_datas.reshape(-1, 90000)
    #
    #
    # # 学習済みモデルの作成
    # fitted_model = load_fitted_model("fitted_model_svc.pickle")
    # output_submit(test_reshaped_data, fitted_model)
    print("end.")

main()

def fit_image_model(X, y, estimator):
    pipe = make_image_pipe_line(estimator)
    return pipe.fit(X, y)


def test_show_image():
    start_time = time.time()
    image_datas = load_image_datas(1)
    top_train_data = image_datas[0]
    print(top_train_data.reshape(-1, 1).transpose().shape)
    Image._show(Image.fromarray(top_train_data))

    # print(top_train_data)
    print("時間 = {:.2f}秒".format(time.time() - start_time))
