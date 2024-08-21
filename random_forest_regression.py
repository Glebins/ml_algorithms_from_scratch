import numpy as np
import pandas as pd

import random

from decision_tree_regression import MyTreeReg


class MyForestReg:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42,
                 max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 oob_score=None, is_parallel_fit=True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.oob_metric = oob_score
        self.is_parallel_fit = is_parallel_fit
        self.oob_score_ = 0

        self.trees = []
        self.leafs_cnt = 0
        self.fi = {}

    def pre_fit(self, X: pd.DataFrame):
        random.seed(self.random_state)
        self.trees.clear()
        self.fi.clear()

        for col_i in X:
            self.fi[col_i] = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit(X)

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(len(X))
        y_index_sorted.index = range(len(y))

        oob_df = pd.DataFrame(0, index=X_index_sorted.index, columns=['value', 'count'])

        for i in range(self.n_estimators):
            cols_idx = random.sample(range(len(X_index_sorted.columns.values.tolist())), round(X_index_sorted.shape[1] * self.max_features))
            rows_idx = random.sample(range(X_index_sorted.shape[0]), round(X_index_sorted.shape[0] * self.max_samples))

            tree_i = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                               max_leafs=self.max_leafs, bins=self.bins)

            X_train_i = X_index_sorted.iloc[rows_idx, cols_idx]
            y_train_i = y_index_sorted.iloc[rows_idx]

            tree_i.fit(X_train_i, y_train_i)

            self.trees.append(tree_i)
            self.leafs_cnt += tree_i.leafs_cnt

            for col_i in self.fi.keys():
                self.fi[col_i] += (tree_i.fi[col_i] if col_i in tree_i.fi else 0) * len(X_train_i) / len(X)

            X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(rows_idx), cols_idx]
            prediction_oob = tree_i.predict(X_oob_i)

            oob_df.loc[X_oob_i.index.values, 'value'] += prediction_oob
            oob_df.loc[X_oob_i.index.values, 'count'] += 1

        oob_prediction_mean = oob_df['value'] / oob_df['count']
        oob_prediction_mean = oob_prediction_mean.dropna()
        y_index_sorted = y_index_sorted[oob_prediction_mean.index]
        self.oob_score_ = self.__get_metric_value(oob_prediction_mean, y_index_sorted)

    def predict(self, X: pd.DataFrame):
        predictions = []

        for tree_i in self.trees:
            predictions.append(tree_i.predict(X))

        return pd.Series(np.array(predictions).mean(axis=0))

    def __get_metric_value(self, y_hat, y):
        if self.oob_metric == 'mae':
            metric_value = (y_hat - y).abs().sum() / len(y)
        elif self.oob_metric == 'mse':
            metric_value = np.square(y_hat - y).sum() / len(y)
        elif self.oob_metric == 'rmse':
            metric_value = np.sqrt(np.square(y_hat - y).sum() / len(y))
        elif self.oob_metric == 'mape':
            metric_value = 100 / len(y) * ((y_hat - y) / y).abs().sum()
        elif self.oob_metric == 'r2':
            metric_value = 1 - (np.square(y - y_hat)).sum() / (np.square(y - y.mean())).sum()
        else:
            metric_value = None

        return metric_value

    def __str__(self):
        res_str = f"{self.__class__.__name__} class: "
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key}={value}, "

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
