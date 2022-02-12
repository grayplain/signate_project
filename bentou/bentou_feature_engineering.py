import numpy as np
import pandas as pd

class Bentou_Name_OneHot:
    def generate_one_hot_bentou_name(self, df):
        keywords = ["白身魚", "カレー", "カツ", "かつ", "チキン", "鶏", "ハンバーグ", "カレイ"]
        countArrays = self.__array_keyword_if_contain(df['name'].values, keywords)

        return pd.DataFrame(countArrays.T, columns=keywords, index=df.index)

    def __array_keyword_if_contain(self, bentouArray, keywords):
        countArrays = []
        for keyword in keywords:
            countArray = []
            for name in bentouArray:
                if keyword in name:
                    countArray.append(1)
                else:
                    countArray.append(0)
            countArrays.append(countArray)

        retValue = np.array(countArrays)
        return retValue