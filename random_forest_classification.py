import numpy as np
import pandas as pd

import multiprocessing
import random

from decision_tree_classification import MyTreeClf


class MyForestClf:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42,
                 max_depth=5, min_samples_split=2, max_leafs=20, bins=16, criterion='entropy',
                 oob_score=None, is_parallel_fit=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion

        self.is_parallel_fit = is_parallel_fit
        self.oob_metric_type = oob_score
        self.oob_score_ = 0

        self.fi = {}
        self.trees = []
        self.leafs_cnt = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.is_parallel_fit:
            self.fit_parallel(X, y)
        else:
            self.fit_sequential(X, y)

    def predict_proba(self, X: pd.DataFrame):
        predictions = []

        for tree_i in self.trees:
            predictions.append(tree_i.predict_proba(X))

        return pd.Series(np.array(predictions).mean(axis=0))

    def predict(self, X: pd.DataFrame, type):
        if type == 'mean':
            return (self.predict_proba(X) > 0.5) * 1
        elif type == 'vote':
            predictions = []

            for tree_i in self.trees:
                predictions.append(tree_i.predict(X))

            predictions = np.array(predictions)
            predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=0, arr=predictions)

            return pd.Series((predictions[1] >= predictions[0]) * 1)
        else:
            raise ValueError('unknown type parameter')

    def fit_parallel(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit(X)

        num_cores = multiprocessing.cpu_count()
        data = (X, y)
        oob_df = self._fit_trees_in_parallel(data, num_cores)

        y_index_sorted = y.copy()
        y_index_sorted.index = range(len(y))

        self.post_fit(oob_df, y_index_sorted)

    def fit_sequential(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit(X)

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(len(X))
        y_index_sorted.index = range(len(y))

        oob_df = pd.DataFrame(0, index=X_index_sorted.index, columns=['value', 'count'])

        for i in range(self.n_estimators):
            cols_idx = random.sample(range(len(X_index_sorted.columns.values.tolist())),
                                     round(X_index_sorted.shape[1] * self.max_features))
            rows_idx = random.sample(range(X_index_sorted.shape[0]), round(X_index_sorted.shape[0] * self.max_samples))

            tree_i = MyTreeClf(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                               max_leafs=self.max_leafs, bins=self.bins, criterion=self.criterion)

            X_train_i = X_index_sorted.iloc[rows_idx, cols_idx]
            y_train_i = y_index_sorted.iloc[rows_idx]

            tree_i.fit(X_train_i, y_train_i)

            self.trees.append(tree_i)
            self.leafs_cnt += tree_i.leafs_cnt

            for col_i in self.fi.keys():
                self.fi[col_i] += (tree_i.fi[col_i] if col_i in tree_i.fi else 0) * len(X_train_i) / len(X)

            X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(rows_idx), cols_idx]
            prediction_oob = tree_i.predict_proba(X_oob_i)

            oob_df.loc[X_oob_i.index.values, 'value'] += prediction_oob
            oob_df.loc[X_oob_i.index.values, 'count'] += 1

        self.post_fit(oob_df, y_index_sorted)

    def pre_fit(self, X: pd.DataFrame):
        self.leafs_cnt = 0
        if self.random_state is not None:
            random.seed(self.random_state)
        self.trees.clear()
        self.fi.clear()

        for col_i in X:
            self.fi[col_i] = 0

    def post_fit(self, oob_df: pd.DataFrame, y_index_sorted):
        oob_prediction_mean = oob_df['value'] / oob_df['count']
        oob_prediction_mean = oob_prediction_mean.dropna()
        y_index_sorted = y_index_sorted[oob_prediction_mean.index]
        self.oob_score_ = self.__get_metric_value(oob_prediction_mean, y_index_sorted)

    def _fit_trees_in_parallel(self, data, num_cores):
        with multiprocessing.Pool(num_cores) as pool:
            models = pool.map(self._fit_tree, [data] * self.n_estimators)
            oob_df = self._combine_parameters(data, models)
        return oob_df

    def _fit_tree(self, data):
        X, y = data

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(len(X))
        y_index_sorted.index = range(len(y))

        cols_idx = random.sample(range(len(X_index_sorted.columns.values.tolist())),
                                 round(X_index_sorted.shape[1] * self.max_features))
        rows_idx = random.sample(range(X_index_sorted.shape[0]), round(X_index_sorted.shape[0] * self.max_samples))

        tree_i = MyTreeClf(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           max_leafs=self.max_leafs, bins=self.bins, criterion=self.criterion)

        X_train_i = X_index_sorted.iloc[rows_idx, cols_idx]
        y_train_i = y_index_sorted.iloc[rows_idx]

        tree_i.fit(X_train_i, y_train_i)

        fi_i = {}
        for col_i in self.fi.keys():
            fi_i[col_i] = (tree_i.fi[col_i] if col_i in tree_i.fi else 0) * len(X_train_i) / len(X)

        X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(rows_idx), cols_idx]
        prediction_oob = tree_i.predict(X_oob_i)

        return tree_i, X_oob_i.index.values, prediction_oob, fi_i

    def _combine_parameters(self, data, models):
        X, y = data
        trees, oob_indices_values, predictions_oob, fi_s = [[model[i] for model in models] for i in range(4)]
        for tree in trees:
            self.leafs_cnt += tree.leafs_cnt
        self.trees = trees

        for fi_i in fi_s:
            for col_i in self.fi.keys():
                self.fi[col_i] += fi_i[col_i]

        oob_df = pd.DataFrame(0, index=range(len(X)), columns=['value', 'count'])

        for index_values_i, prediction_oob_i in zip(oob_indices_values, predictions_oob):
            oob_df.loc[index_values_i, 'value'] += prediction_oob_i
            oob_df.loc[index_values_i, 'count'] += 1

        return oob_df

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

        if self.oob_metric_type == 'accuracy':
            return pd.Series(y_labels == y_original).mean()
        elif self.oob_metric_type == 'precision':
            return TP / (TP + FP)
        elif self.oob_metric_type == 'recall':
            return TP / (TP + FN)
        elif self.oob_metric_type == 'f1':
            return 2 * TP / (2 * TP + FP + FN)
        elif self.oob_metric_type == 'roc_auc':
            scores = pd.DataFrame()
            scores['probability'] = y_predicted.round(10)
            scores['truth'] = y_original
            scores = scores.sort_values(by='probability', ascending=False)
            scores['roc_auc_param'] = scores.apply(self.calculate_roc_auc_column_helper, axis=1, args=(scores,))

            negative_classes_number, positive_classes_number = [scores['truth'].value_counts()[i] for i in range(2)]
            roc_auc_sum = scores['roc_auc_param'].sum()

            return roc_auc_sum / (negative_classes_number * positive_classes_number)
        elif self.oob_metric_type is None:
            return None
        else:
            raise ValueError('Unknown type of metric. Use one of the following: accuracy, precision, recall, f1, '
                             'roc_auc')

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
