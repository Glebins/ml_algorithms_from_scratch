import numpy as np
import pandas as pd


def get_entropy_of_col(column: pd.Series):
    len_objects = len(column)

    unique_objects = column.value_counts()

    entropy = 0
    for item, item_repetitions in unique_objects.items():
        p_i = item_repetitions / len_objects
        if p_i == 0:
            entropy -= 0
        else:
            entropy -= p_i * np.log2(p_i)

    return entropy


def get_best_split(X: pd.DataFrame, y: pd.Series):

    best_params = {'col_name': None, 'split_value': None, 'ig': 0}

    for column_i in X:
        delimiters = X[column_i].unique()

        len_objects = len(X[column_i])

        S_0 = get_entropy_of_col(X[column_i])

        for delim in delimiters:
            basket_left = X[X[column_i] <= delim][column_i]
            basket_right = X[X[column_i] > delim][column_i]

            if basket_right.empty or basket_left.empty:
                continue

            S_1 = get_entropy_of_col(basket_left)
            S_2 = get_entropy_of_col(basket_right)

            information_gain = S_0 - len(basket_left) / len_objects * S_1 - len(basket_right) / len_objects * S_2

            print(basket_left.tolist(), basket_right.tolist(), delim, S_0, S_1, S_0 - S_1 * len(basket_left) / len_objects)

            if information_gain > best_params['ig']:
                best_params['ig'] = information_gain
                best_params['col_name'] = column_i
                best_params['split_value'] = delim

    return best_params['col_name'], best_params['split_value'], best_params['ig']


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self):
        res_str = f"{self.__class__.__name__} class: "
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key} = {value}, "

        res_str = res_str[:-2]
        return res_str

    def __repr__(self):
        res_str = f"{self.__class__.__name__}("

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key}={value}, "

        res_str = res_str[:-2]
        res_str += ")"

        return res_str
