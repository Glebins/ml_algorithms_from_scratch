import pandas as pd
import numpy as np


def add_column_of_ones(X: pd.DataFrame):
    X1 = X.copy()
    X1.insert(0, -1, 1)
    X1.columns = pd.RangeIndex(0, len(X1.columns.values), 1)
    return X1


def generate_weights(size):
    return pd.Series([1] * size)  # naive implementation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def construct_X(X):
    return pd.DataFrame(data=np.array(X))


def construct_y(y):
    return pd.Series(y)


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric

        self.weights = None

        self.__loss = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        dimension_size = len(X.columns.values)
        number_points = len(X.index.values)

        X = add_column_of_ones(X)

        X_original = X.copy()
        y_original = y.copy()

        self.weights = generate_weights(dimension_size + 1)

        for i in range(1, self.n_iter + 1):
            y_predicted = sigmoid(X @ self.weights)

            eps = 1e-15
            self.__loss = - 1 / len(X) * (y * np.log(y_predicted + eps) + (1 - y) * np.log(1 - y_predicted + eps)).sum()

            grad = 1 / len(X) * (y_predicted - y) @ X

            delta_weights = -self.learning_rate * grad

            self.weights += delta_weights

            if verbose and (i == 1 or i % verbose == 0 or i == self.n_iter):
                # self.__metric_value = self.__get_metric_value(X @ self.weights, y_original)
                self.__debug_while_fit(i)

        # y_predicted = X @ self.weights
        # self.__metric_value = self.__get_metric_value(y_predicted, y_original)

    def predict_proba(self, X: pd.DataFrame):
        X = add_column_of_ones(X)

        return sigmoid(X @ self.weights)

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X) > 0.5) * 1

    def __debug_while_fit(self, i):
        # metric_part = f"\tmetric_{self.metric_type} = {self.__metric_value}" if self.__metric_value is not None else ''
        print(f"{i} out of {self.n_iter}\tloss = {self.__loss}")

    def get_coef(self):
        return self.weights[1:]

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
