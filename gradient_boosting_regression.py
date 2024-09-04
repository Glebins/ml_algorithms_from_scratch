import numpy as np
import pandas as pd

from decision_tree_regression import *


class MyBoostReg:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20,
                 bins=16, loss='MSE'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_type = loss

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = None
        self.trees = []

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        self.pred_0 = y.mean()

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(X.shape[0])
        y_index_sorted.index = range(len(y))

        prev_pred = pd.Series([self.pred_0] * len(y))
        sum_pred_predictions = prev_pred

        for i in range(self.n_estimators):
            tree_i = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                               max_leafs=self.max_leafs, bins=self.bins)

            remainder = y_index_sorted - sum_pred_predictions

            tree_i.fit(X_index_sorted, remainder)

            prev_pred = pd.Series(tree_i.predict(X_index_sorted))
            sum_pred_predictions += self.learning_rate * prev_pred

            self.trees.append(tree_i)

    def predict(self, X: pd.Series):
        result = pd.Series([self.pred_0] * X.shape[0])

        for tree_i in self.trees:
            result += self.learning_rate * pd.Series(tree_i.predict(X))

        return result

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
