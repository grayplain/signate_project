from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
import time


from sklearn.preprocessing import StandardScaler
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
        images_list.append(img)

    images = np.asarray(images_list)
    return images

def main():
    start_time = time.time()
    image_datas = load_image_datas(1)
    top_train_data = image_datas[0]

    print(top_train_data.reshape(-1, 1).transpose().shape)
    Image._show(Image.fromarray(top_train_data))

    # print(top_train_data)
    print("時間 = {:.2f}秒".format(time.time() - start_time))
    print("end.")

main()