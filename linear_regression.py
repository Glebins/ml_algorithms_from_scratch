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


def construct_X(X):
    return pd.DataFrame(data=np.array(X).T)


def construct_y(y):
    return pd.Series(y)


class LinearReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42, train_X=None, train_y=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric_type = metric
        self.reg_type = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample if sgd_sample is not None else 1.0
        self.random_state = random_state

        if train_X is not None:
            train_X = add_column_of_ones(train_X)

        self.__train_X = train_X
        self.__train_y = train_y

        self.weights = None
        self.__metric_value = None
        self.__loss = 0

    def fit(self, verbose=False):
        random.seed(self.random_state)

        dimension_size = len(self.__train_X.columns.values)
        number_points = len(self.__train_X.index.values)

        X_original = self.__train_X.copy()
        y_original = self.__train_y.copy()

        self.weights = generate_weights(dimension_size)

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
                self.__metric_value = self.__get_metric_value(X @ self.weights, y_original)
                self.__debug_while_fit(i)

        y_predicted = self.__train_X @ self.weights
        self.__metric_value = self.__get_metric_value(y_predicted, y_original)

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

    def __get_metric_value(self, y_hat, y):
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

    def predict(self):
        return self.__train_X @ self.weights

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

    def get_solution(self):
        tmp = self.__train_X.T @ self.__train_X
        inv_tmp = pd.DataFrame(np.linalg.inv(tmp), tmp.columns, tmp.index)
        beta = inv_tmp @ self.__train_X.T @ self.__train_y
        self.weights = beta

        return beta

    def get_metric(self):
        y_hat = self.__train_X @ self.weights

        return self.metric_type, self.__get_metric_value(y_hat, self.__train_y)

    def get_pretty_string_of_points(self, in_a_row=True, include_zero=False, columns=None):
        result = ('[' if in_a_row else '')
        delimiter = (', ' if in_a_row else '\n')
        end_symbol = (']' if in_a_row else '')

        for i in range(self.__train_X.shape[0]):
            result += '('
            # end_of_cycle = 2 if is_one_variable else self.__train_X.shape[1]
            cols_to_iterate = (range(self.__train_X.shape[1]) if include_zero else range(1, self.__train_X.shape[1])) \
                if columns is None else columns
            for j in cols_to_iterate:
                result += str(self.__train_X.iloc[i, j])
                result += ', '
            result += str(self.__train_y[i]) + ')' + delimiter
        result = result[:-len(delimiter)]
        result += end_symbol

        return result

    def get_pretty_str_of_result(self, variables=None, round_to=None):
        if variables is None:
            if len(self.weights) == 2:
                variables = ['y', 'x']
            elif len(self.weights) == 3:
                variables = ['z', 'x', 'y']
            else:
                variables = ['y'] + [f'x{i}' for i in range(len(self.weights))]

        rounded_weights = round(self.weights, round_to) if round_to is not None else self.weights

        tmp = ([f"{coefficient} * {variable}" for coefficient, variable in zip(rounded_weights[1:], variables[1:])] +
               [str(rounded_weights[0])])

        res = f"{variables[0]} = {' + '.join(tmp)}"

        return res

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

    @property
    def train_X(self):
        return self.__train_X

    @train_X.setter
    def train_X(self, value):
        if value is not None:
            value = add_column_of_ones(value)
        self.__train_X = value

    @property
    def train_y(self):
        return self.__train_y

    @train_y.setter
    def train_y(self, value):
        self.__train_y = value

    def get_coef(self):
        return self.weights.to_numpy()[1:]

    def get_weights(self):
        return self.weights
