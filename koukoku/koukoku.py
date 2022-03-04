import numpy as np
import pandas as pd

import os


def file_path(file_name):
    return 'koukoku' + '/' + file_name

# 提出ファイル作成
def output_submit(result, file_name='submit_blank.csv'):
    sample_submit = read_pd_data(file_path('sample_submit.csv'), headerFlag=None)
    sample_submit[1] = result
    sample_submit.to_csv(file_name, header=None, sep=',')


def read_pd_data(file_name, headerFlag="infer"):
    return pd.read_table(os.getcwd() + '/datas/' + file_name, index_col=0, header=headerFlag)





def main():
    train_data = read_pd_data("train.tsv")
    # train_data = read_pd_data("train.tsv")
    print(train_data.min())
    print("----------------")
    print(train_data.max())
    print("----------------")
    print(train_data.median())
    print("----------------")
    print(train_data.mode())
    print("----------------")

main()


