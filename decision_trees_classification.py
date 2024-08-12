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

    S_0 = get_entropy_of_col(y)

    for column_i in X:
        unique_values = sorted(X[column_i].unique())
        delimiters = []
        for i in range(len(unique_values) - 1):
            mean_i = (unique_values[i] + unique_values[i + 1]) / 2
            delimiters.append(mean_i)

        len_objects = len(X[column_i])

        for delim in delimiters:
            X_left = X[X[column_i] <= delim][column_i]
            X_right = X[X[column_i] > delim][column_i]

            y_left = y[X_left.index]
            y_right = y[X_right.index]

            if X_left.empty or X_right.empty:
                continue

            S_1 = get_entropy_of_col(y_left)
            S_2 = get_entropy_of_col(y_right)

            information_gain = S_0 - len(y_left) / len_objects * S_1 - len(y_right) / len_objects * S_2

            print(delim, y_left.tolist(), y_right.tolist(), column_i, information_gain)

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

        self.leafs_cnt = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        col_name, split_value, _ = get_best_split(X, y)

        X_left = X[X[col_name] <= split_value]
        X_right = X[X[col_name] > split_value]

        y_left = y[X_left.index]
        y_right = y[X_right.index]



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


class Tree:
    def __init__(self, col_name=None, split_value=None):
        self.left = None
        self.right = None

        self.col_name = col_name
        self.split_value = split_value

    def add_node(self, is_left: bool):
        if is_left:
            self.left = Tree()
        else:
            self.right = Tree()

    @property
    def col_name(self):
        return self.col_name

    @col_name.setter
    def col_name(self, col_name_new):
        self.col_name = col_name_new

    @property
    def split_value(self):
        return self.split_value

    @split_value.setter
    def split_value(self, split_value_new):
        self.split_value = split_value_new
