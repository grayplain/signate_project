from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
import time
import cv2


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# 提出ファイル作成
def output_submit(test_data, estimator):
    sample_submit = read_pd_data('sample_submit.tsv', header=None)
    pred = estimator.predict(test_data)
    sample_submit[1] = pred
    sample_submit.to_csv('submit_number.tsv', header=None, sep='\t')

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


def open_image(file_name, is_train_data):
    env = 'train_' if is_train_data else 'test_'

    file_name_path = env + 'data/' + file_name
    return np.array(Image.open(os.getcwd() + '/datas/' + file_name_path))


def load_image_datas(max_number, is_train_data=True):
    pixel = 300*300
    env = 'train_data_' if is_train_data else 'test_'

    images = np.empty((0, pixel))
    images_list = images.tolist()

    if max_number < 0:
        print('0未満の値の画像は存在しません。')
        return

    if max_number > 250:
        print('とりあえず。画像水増ししたらこの処理は不要。 ')
        return


    for num in range(max_number):
        file_name = env + str(num + 1) + '.jpeg'
        img = open_image(file_name, is_train_data)
        # グレースケール化
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images_list.append(gray_scale)


    images = np.asarray(images_list)
    return images


def classifier_number_from_images(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    # estimators = [RidgeClassifier()]
    estimators = [RidgeClassifier(), SVC(), RandomForestClassifier()]


    # return
    print('全データ数 = {}'.format(X.shape[0]))
    for estimator in estimators:
        print('推定器 =           {}'.format(estimator.__class__))
        pipe = make_image_pipe_line(estimator)
        print('結果(推定器単体) =    {:.2f}'.format(estimator.fit(X_train, y_train).score(X_test, y_test)))
        print('結果(パイプ) =       {:.2f}'.format(pipe.fit(X_train, y_train).score(X_test, y_test)))
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
    data_count = 250

    # 画像データ読み込み
    image_datas = load_image_datas(data_count)
    reshaped_data = image_datas.reshape(-1, 90000)

    # 正解データの読み込み
    num_pd = read_pd_data('train.csv')
    y = num_pd[:data_count]['target'].values

    # 予測
    classifier_number_from_images(reshaped_data, y)
    print("end.")

main()


def test_show_image():
    start_time = time.time()
    image_datas = load_image_datas(1)
    top_train_data = image_datas[0]
    print(top_train_data.reshape(-1, 1).transpose().shape)
    Image._show(Image.fromarray(top_train_data))

    # print(top_train_data)
    print("時間 = {:.2f}秒".format(time.time() - start_time))
