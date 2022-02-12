import numpy as np
import pandas as pd

class Bentou_Name_OneHot:
    def generate_one_hot_bentou_name(self, df):
        keywords = ["白身魚",
                    "バーベキュー",
                     "ポーク", "豚",
                    "マスタード",
                    "トマト", "野菜",
                    "丼",
                    "キムチ",
                    "炒め",
                    "唐揚げ", "唐揚",
                    "生姜",
                    "フライ",
                    "焼き",
                    "チンジャオ",
                    "豆腐",
                    "味噌", "みそ",
                    "カレー",
                    "メンチ",
                    "カツ", "かつ",  # これ同じメニューになるので、同じメニューにする処理が必要かも。
                    "チキン", "鶏",
                    "ハンバーグ",
                    "カレイ"]
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