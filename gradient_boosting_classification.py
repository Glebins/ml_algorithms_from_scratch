import random

import numpy as np
import pandas as pd

from decision_tree_regression import *


class MyBoostClf:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_features=0.5, max_samples=0.5, random_state=42,
                 reg=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.metric_type = metric
        self.reg_coef = reg

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = None
        self.trees = []
        self.best_score = 0
        self.fi = {}

        self.__eps = 1e-15

        self.__test_metric_types()

    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        self.pred_0 = self.calculate_log_odds(y)
        self.best_score = 0
        self.trees.clear()

        if self.random_state:
            random.seed(self.random_state)

        self.fi.clear()

        for col_i in X:
            self.fi[col_i] = 0.0
        doing_early_stopping = self.__test_early_stopping(X_eval, y_eval, early_stopping)

        train_index = X.index
        validation_index = X_eval.index if doing_early_stopping else None

        X.index = range(X.shape[0])
        y.index = range(len(y))

        steps_of_non_improvement_valid = 0
        last_best_score = 0 if self.metric_type is not None else float('inf')

        if doing_early_stopping:
            X_eval.index = range(X_eval.shape[0])
            y_eval.index = range(len(y_eval))

        current_predict = pd.Series([self.pred_0] * len(y))
        eval_predict = pd.Series([self.pred_0] * len(y_eval)) if doing_early_stopping else None
        sum_prev_predictions = current_predict

        loss_value = 0
        current_amount_of_leaves = 0

        for i in range(1, self.n_estimators + 1):
            tree_i = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                               max_leafs=self.max_leafs, bins=self.bins)

            cols_idx = random.sample(range(len(X.columns.values.tolist())),
                                     round(X.shape[1] * self.max_features))
            rows_idx = random.sample(range(X.shape[0]),
                                     round(X.shape[0] * self.max_samples))

            rows_idx.sort()
            cols_idx.sort()

            X_train_i = X.iloc[rows_idx, cols_idx]
            X_train_i.index = range(X_train_i.shape[0])
            y_train_i = y.iloc[rows_idx]
            y_train_i.index = range(y_train_i.shape[0])

            pred_prob = self.get_probability(sum_prev_predictions)
            pred_prob_i = pred_prob.iloc[rows_idx]
            pred_prob_i.index = range(pred_prob_i.shape[0])

            grad = pred_prob - y
            grad_part = grad.iloc[rows_idx]
            grad_part.index = range(grad_part.shape[0])

            tree_i.fit(X_train_i, -grad_part, do_write_elements_in_leaves=True)

            addition = self.reg_coef * current_amount_of_leaves
            current_amount_of_leaves += tree_i.leafs_cnt

            self.recalculate_leaves_values(tree_i.tree, tree_i.get_elements_in_leaves(), y_train_i,
                                           pred_prob_i, addition)

            current_predict = pd.Series(tree_i.predict(X))

            if callable(self.learning_rate):
                sum_prev_predictions += self.learning_rate(i) * current_predict
            else:
                sum_prev_predictions += self.learning_rate * current_predict

            self.trees.append(tree_i)

            for col_i in self.fi.keys():
                self.fi[col_i] += (tree_i.fi[col_i] if col_i in tree_i.fi else 0) * len(X_train_i) / len(X)

            loss_value = self.calculate_log_loss(sum_prev_predictions, y)

            metric_value = None
            if self.metric_type is not None:
                metric_value = self.calculate_metric(sum_prev_predictions, y, self.metric_type)

            validation_value = None
            if doing_early_stopping:
                current_eval_prediction = pd.Series(tree_i.predict(X_eval))

                if callable(self.learning_rate):
                    eval_predict += self.learning_rate(i) * current_eval_prediction
                else:
                    eval_predict += self.learning_rate * current_eval_prediction

                if self.metric_type:
                    validation_value = self.calculate_metric(eval_predict, y_eval, self.metric_type)
                else:
                    validation_value = self.calculate_log_loss(eval_predict, y_eval)

                validation_metric = self.metric_type if self.metric_type is not None else "LogLoss"

                if self.__is_non_improvement(last_best_score, validation_value, validation_metric):
                    steps_of_non_improvement_valid += 1
                else:
                    last_best_score = validation_value
                    steps_of_non_improvement_valid = 0

                if steps_of_non_improvement_valid == early_stopping:
                    if verbose is not None:
                        print(f"---------------Validation = {validation_value}, early stopping---------------")
                    del self.trees[-early_stopping:]
                    self.best_score = last_best_score
                    break

            if verbose is not None and (i % verbose == 0 or i == self.n_estimators):
                self.__debug_while_fit(i, loss_value, metric_value, validation_value)

        else:
            if self.metric_type is not None:
                self.best_score = self.calculate_metric(sum_prev_predictions, y, self.metric_type)
            else:
                self.best_score = loss_value

        X.index = train_index
        y.index = train_index
        if validation_index is not None:
            X_eval.index = validation_index
            y_eval.index = validation_index

    def predict_proba(self, X: pd.DataFrame):
        sum_prev_predictions = pd.Series([self.pred_0] * X.shape[0])

        for i, tree_i in enumerate(self.trees):
            pred_i = pd.Series(tree_i.predict(X))

            if callable(self.learning_rate):
                sum_prev_predictions += self.learning_rate(i + 1) * pred_i
            else:
                sum_prev_predictions += self.learning_rate * pred_i

        return self.get_probability(sum_prev_predictions)

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X) > 0.5) * 1

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

    @staticmethod
    def calculate_metric(y_predicted: pd.Series, y_original: pd.Series, metric_type):
        y_predicted = MyBoostClf.get_probability(y_predicted)
        y_labels = pd.Series((y_predicted > 0.5) * 1)
        TP = pd.Series(y_original[y_original == 1] == y_labels[y_original == 1]).sum()
        TN = pd.Series(y_original[y_original == 0] == y_labels[y_original == 0]).sum()
        FP = pd.Series(y_original[y_original == 0] != y_labels[y_original == 0]).sum()
        FN = pd.Series(y_original[y_original == 1] != y_labels[y_original == 1]).sum()

        if metric_type == 'accuracy':
            return pd.Series(y_labels == y_original).mean()
        elif metric_type == 'precision':
            return TP / (TP + FP)
        elif metric_type == 'recall':
            return TP / (TP + FN)
        elif metric_type == 'f1':
            return 2 * TP / (2 * TP + FP + FN)
        elif metric_type == 'roc_auc':
            scores = pd.DataFrame()
            scores['probability'] = y_predicted
            scores['truth'] = y_original
            scores = scores.sort_values(by='probability', ascending=False)
            scores['roc_auc_param'] = scores.apply(MyBoostClf.calculate_roc_auc_column_helper,
                                                   axis=1, args=(scores,))

            negative_classes_number, positive_classes_number =\
                [scores['truth'].value_counts()[i] for i in range(2)]
            roc_auc_sum = scores['roc_auc_param'].sum()

            return roc_auc_sum / (negative_classes_number * positive_classes_number)
        elif metric_type is None:
            return None
        else:
            raise ValueError("Unknown metric (or it's been changed?)")

    def calculate_log_loss(self, y_predicted: pd.Series, y_original: pd.Series):
        return -(y_original * y_predicted - np.log(1 + np.exp(y_predicted) + self.__eps)).mean()

    def recalculate_leaves_values(self, tree, indices_elements_in_leaves, y, p, addition, i=0):
        if tree.depth == 0:
            i = 0

        if not tree.is_leaf:
            i = self.recalculate_leaves_values(tree.left, indices_elements_in_leaves, y, p, addition, i)
            i = self.recalculate_leaves_values(tree.right, indices_elements_in_leaves, y, p, addition, i)
        else:
            y_i = y.iloc[indices_elements_in_leaves[i]]
            p_i = p.iloc[indices_elements_in_leaves[i]]
            tree.leaf_value = (y_i - p_i).sum() / (p_i * (1 - p_i)).sum() + addition
            i += 1

        if tree.depth != 0:
            return i

    def calculate_log_odds(self, y: pd.Series):
        y_mean = y.mean()
        return np.log(y_mean / (1 - y_mean) + self.__eps)

    @staticmethod
    def get_probability(y: pd.Series):
        return np.exp(y) / (1 + np.exp(y))

    def __debug_while_fit(self, i, loss_value, metric_value, validation_value):
        metric_part = ""
        validation_part = ""
        validation_metric = self.metric_type if self.metric_type else "LogLoss"
        if metric_value is not None:
            metric_part = f",\tMetric({self.metric_type}) = {metric_value}"
        if validation_value is not None:
            validation_part = f",\tValidation({validation_metric}) = {validation_value}"
        print(f"{i}. Loss = {loss_value}" + metric_part + validation_part)

    def __test_metric_types(self):
        metric_types = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')

        if self.metric_type not in metric_types and self.metric_type is not None:
            raise ValueError(f"Metric type must be one of this: {metric_types}")

    @staticmethod
    def __is_non_improvement(prev_val, current_val, metric):
        if metric in ('LogLoss',):
            return current_val >= prev_val
        else:
            return current_val <= prev_val

    @staticmethod
    def __test_early_stopping(X_eval, y_eval, early_stopping):
        if X_eval is not None and y_eval is not None and early_stopping is not None:
            return True
        else:
            return False

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
