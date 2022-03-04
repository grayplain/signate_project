import numpy as np
from numpy.random import *
import pandas as pd
import datetime
import math

import os


# 提出ファイル作成
def output_submit(result, file_name='submit_blank.csv'):
    sample_submit = read_pd_data(file_path('sample_submit.csv'), headerFlag=None)
    sample_submit[1] = result
    sample_submit.to_csv(file_name, header=None, sep=',')


def read_pd_data(file_name, headerFlag="infer"):
    return pd.read_csv(os.getcwd() + '/datas/' + file_name, index_col=0, header=headerFlag)





def main():
    train_data = read_pd_data("train.tsv")
    


main()


