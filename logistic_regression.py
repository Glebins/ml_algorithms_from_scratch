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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def construct_X(X):
    return pd.DataFrame(data=np.array(X))


def construct_y(y):
    return pd.Series(y)


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=None, l2_coef=None, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric_type = metric
        self.regularization_type = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample if sgd_sample is not None else 1.0
        self.random_state = random_state

        self.weights = None

        self.__metric_value = 0
        self.__loss = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)

        dimension_size = len(X.columns.values)
        number_points = len(X.index.values)

        X = add_column_of_ones(X)

        X_original = X.copy()
        y_original = y.copy()

        self.weights = generate_weights(dimension_size + 1)

        for i in range(1, self.n_iter + 1):
            X, y = self.__get_batch(X_original, y_original)
            y_predicted = sigmoid(X @ self.weights)

            eps = 1e-15
            self.__loss = (- 1 / len(X_original) * (y_original * np.log(y_predicted + eps)
                                                    + (1 - y_original) * np.log(1 - y_predicted + eps)).sum()
                           + self.calculate_reg_loss())

            grad = 1 / len(X) * (y_predicted - y) @ X + self.calculate_reg_grad()

            if callable(self.learning_rate):
                delta_weights = -self.learning_rate(i) * grad
            else:
                delta_weights = -self.learning_rate * grad

            self.weights += delta_weights

            if verbose and (i == 1 or i % verbose == 0 or i == self.n_iter):
                self.__metric_value = self.__get_metric_value(self.predict_proba(X, do_add_ones=False), y)
                self.__debug_while_fit(i)

        self.__metric_value = self.__get_metric_value(self.predict_proba(X, do_add_ones=False), y)

    def calculate_reg_loss(self):
        if self.regularization_type == 'l1':
            return self.l1_coef * self.weights.abs().sum()
        elif self.regularization_type == 'l2':
            return self.l2_coef * np.square(self.weights).sum()
        elif self.regularization_type == 'elasticnet':
            return self.l1_coef * self.weights.abs().sum() + self.l2_coef * np.square(self.weights).sum()
        elif self.regularization_type is None:
            return 0
        else:
            raise ValueError(f'Unknown regularization parameter: {self.regularization_type}, use l1, l2 or elasticnet')

    def calculate_reg_grad(self):
        if self.regularization_type == 'l1':
            return self.l1_coef * np.sign(self.weights)
        elif self.regularization_type == 'l2':
            return self.l2_coef * 2 * self.weights
        elif self.regularization_type == 'elasticnet':
            return self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        elif self.regularization_type is None:
            return 0
        else:
            raise ValueError(f'Unknown regularization parameter: {self.regularization_type}, use l1, l2 or elasticnet')

    def predict_proba(self, X: pd.DataFrame, do_add_ones=True):
        if do_add_ones:
            X = add_column_of_ones(X)

        return sigmoid(X @ self.weights)

    def predict(self, X: pd.DataFrame, do_add_ones=True):
        return (self.predict_proba(X, do_add_ones) > 0.5) * 1

    @staticmethod
    def calculate_roc_auc_column_helper(row, scores):
        score_i = row[0]
        ground_truth_i = row[1]

        if ground_truth_i == 1:
            return 0

        num_positive_scores_bigger = scores[(scores['probability'] > score_i) & (scores['truth'] == 1)].shape[0]
        num_positive_scores_equal = scores[(scores['probability'] == score_i) &
                                           (scores['truth'] == 1)].shape[0]
        return num_positive_scores_bigger + num_positive_scores_equal / 2

    def __get_metric_value(self, y_predicted: pd.Series, y_original: pd.Series):
        y_labels = pd.Series((y_predicted > 0.5) * 1)
        TP = pd.Series(y_original[y_original == 1] == y_labels[y_original == 1]).sum()
        TN = pd.Series(y_original[y_original == 0] == y_labels[y_original == 0]).sum()
        FP = pd.Series(y_original[y_original == 0] != y_labels[y_original == 0]).sum()
        FN = pd.Series(y_original[y_original == 1] != y_labels[y_original == 1]).sum()

        if self.metric_type == 'accuracy':
            return pd.Series(y_labels == y_original).mean()
        elif self.metric_type == 'precision':
            return TP / (TP + FP)
        elif self.metric_type == 'recall':
            return TP / (TP + FN)
        elif self.metric_type == 'f1':
            return 2 * TP / (2 * TP + FP + FN)
        elif self.metric_type == 'roc_auc':
            scores = pd.DataFrame()
            scores['probability'] = y_predicted
            scores['truth'] = y_original
            scores = scores.sort_values(by='probability', ascending=False)
            scores['roc_auc_param'] = scores.apply(self.calculate_roc_auc_column_helper, axis=1, args=(scores,))

            negative_classes_number, positive_classes_number = [scores['truth'].value_counts()[i] for i in range(2)]
            roc_auc_sum = scores['roc_auc_param'].sum()

            return roc_auc_sum / (negative_classes_number * positive_classes_number)
        elif self.metric_type is None:
            return None
        else:
            raise ValueError('Unknown type of metric. Use one of the following: accuracy, precision, recall, f1, '
                             'roc_auc')

    def __get_batch(self, X, y):
        number_points = len(X.index.values)

        if isinstance(self.sgd_sample, int):
            sample_indexes = random.sample(range(number_points), self.sgd_sample)
        else:
            sample_indexes = random.sample(range(number_points), round(self.sgd_sample * number_points))

        X_ = X.iloc[sample_indexes]
        y_ = y.iloc[sample_indexes]

        return X_, y_

    def get_best_score(self):
        return self.__metric_value

    def __debug_while_fit(self, i):
        metric_part = f"\tmetric_{self.metric_type} = {self.__metric_value}" if self.__metric_value is not None else ''
        print(f"{i} out of {self.n_iter}\tloss = {self.__loss}" + metric_part)

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
