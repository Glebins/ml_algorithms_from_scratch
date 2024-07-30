import pandas as pd
import numpy as np
import random


def add_column_of_ones(X: pd.DataFrame):
    X1 = X.copy()
    X1.insert(0, -1, 1)
    X1.columns = pd.RangeIndex(0, len(X1.columns.values), 1)
    return X1


def generate_weights(size):
    return pd.Series([1] * size)  # naive implementation


class LinearReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric_type = metric
        self.reg_type = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample if sgd_sample is not None else 1.0
        self.random_state = random_state

        self.weights = None
        self.__metric_value = None
        self.__loss = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)

        dimension_size = len(X.columns.values)
        number_points = len(X.index.values)

        X_original = add_column_of_ones(X)
        y_original = y.copy()

        self.weights = generate_weights(dimension_size + 1)

        for i in range(1, self.n_iter + 1):
            X, y = self.__get_batch(X_original, y_original)

            y_predicted = X @ self.weights

            self.__loss = (np.square(X_original @ self.weights - y_original).sum() / number_points +
                           self.calculate_reg_loss())

            grad = 2 * (y_predicted - y) @ X / len(X.index.values) + self.calculate_reg_grad()

            if callable(self.learning_rate):
                delta_weights = -self.learning_rate(i) * grad
            else:
                delta_weights = -self.learning_rate * grad

            self.weights += delta_weights

            if verbose and (i == 1 or i % verbose == 0 or i == self.n_iter):
                self.__metric_value = self.get_metric(X @ self.weights, y_original)
                self.__debug_while_fit(i)

        y_predicted = X @ self.weights
        self.__metric_value = self.get_metric(y_predicted, y_original)

    def calculate_reg_loss(self):
        if self.reg_type == 'l1':
            return self.l1_coef * self.weights.abs().sum()
        elif self.reg_type == 'l2':
            return self.l2_coef * np.square(self.weights).sum()
        elif self.reg_type == 'elasticnet':
            return self.l1_coef * self.weights.abs().sum() + self.l2_coef * np.square(self.weights).sum()
        else:
            return 0

    def calculate_reg_grad(self):
        if self.reg_type == 'l1':
            return self.l1_coef * np.sign(self.weights)
        elif self.reg_type == 'l2':
            return self.l2_coef * 2 * self.weights
        elif self.reg_type == 'elasticnet':
            return self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        else:
            return 0

    def get_metric(self, y_hat, y):
        if self.metric_type == 'mae':
            metric_value = (y_hat - y).abs().sum() / len(y)
        elif self.metric_type == 'mse':
            metric_value = np.square(y_hat - y).sum() / len(y)
        elif self.metric_type == 'rmse':
            metric_value = np.sqrt(np.square(y_hat - y).sum() / len(y))
        elif self.metric_type == 'mape':
            metric_value = 100 / len(y) * ((y_hat - y) / y).abs().sum()
        elif self.metric_type == 'r2':
            metric_value = 1 - (np.square(y - y_hat)).sum() / (np.square(y - y.mean())).sum()
        else:
            metric_value = None

        return metric_value

    def predict(self, X: pd.DataFrame):
        X = add_column_of_ones(X)

        return X @ self.weights

    def get_best_score(self):
        return self.__metric_value if self.metric_type is not None else self.__loss

    def __debug_while_fit(self, i):
        metric_part = f"\tmetric_{self.metric_type} = {self.__metric_value}" if self.__metric_value is not None else ''
        print(f"{i} out of {self.n_iter}\tloss = {self.__loss}" + metric_part)

    def __get_batch(self, X, y):
        number_points = len(X.index.values)

        if isinstance(self.sgd_sample, int):
            sample_indexes = random.sample(range(number_points), self.sgd_sample)
        else:
            sample_indexes = random.sample(range(number_points), round(self.sgd_sample * number_points))

        X_ = X.iloc[sample_indexes]
        y_ = y.iloc[sample_indexes]

        return X_, y_

    @staticmethod
    def get_solution(X: pd.DataFrame, y: pd.Series):
        X = add_column_of_ones(X)
        tmp = X.T @ X
        inv_tmp = pd.DataFrame(np.linalg.inv(tmp), tmp.columns, tmp.index)
        beta = inv_tmp @ X.T @ y

        return beta

    def get_metric_for_ideal_solution(self, X, y):
        self.weights = self.get_solution(X, y)
        X = add_column_of_ones(X)
        y_hat = X @ self.weights

        return f"{self.metric_type} = {self.get_metric(y_hat, y)}"

    def __str__(self):
        res_str = "LinearReg class: "
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key} = {value}, "

        res_str = res_str[:-2]
        return res_str

    def __repr__(self):
        res_str = "LinearReg("

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key}={value}, "

        res_str = res_str[:-2]
        res_str += ")"

        return res_str

    def get_coef(self):
        return self.weights.to_numpy()[1:]

    def get_weights(self):
        return self.weights
