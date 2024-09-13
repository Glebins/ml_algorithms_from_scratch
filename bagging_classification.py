import numpy as np
import pandas as pd
import random
import copy
import multiprocessing


class MyBaggingClf:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42, oob_score=None,
                 is_parallel_fit=True):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.oob_metric_type = oob_score
        self.is_parallel_fit = is_parallel_fit

        self.oob_score_ = 0
        self.estimators = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.is_parallel_fit:
            self.fit_parallel(X, y)
        else:
            self.fit_sequential(X, y)

    def predict_proba(self, X: pd.DataFrame):
        predictions = []

        for model in self.estimators:
            predictions.append(model.predict_proba(X))

        return pd.Series(np.array(predictions).mean(axis=0))

    def predict(self, X: pd.DataFrame, type):
        if type == 'mean':
            return (self.predict_proba(X) > 0.5) * 1
        elif type == 'vote':
            predictions = []

            for model in self.estimators:
                predictions.append(model.predict(X))

            predictions = np.array(predictions)
            predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=0, arr=predictions)

            return pd.Series((predictions[1] >= predictions[0]) * 1)
        else:
            raise ValueError('unknown type parameter')

    def fit_sequential(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit()

        sample_rows = []

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(len(X))
        y_index_sorted.index = range(len(y))

        for i in range(self.n_estimators):
            sample_rows_idx_i = random.choices(range(len(X_index_sorted.index.values.tolist())),
                                               k=round(X_index_sorted.shape[0] * self.max_samples))
            sample_rows.append(sample_rows_idx_i)

        oob_df = pd.DataFrame(0.0, index=X_index_sorted.index, columns=['value', 'count'])

        for i in range(self.n_estimators):
            sample_rows_i = sample_rows[i]
            model_i = copy.deepcopy(self.estimator)

            X_sample = X.iloc[sample_rows_i]
            y_sample = y.iloc[sample_rows_i]

            model_i.fit(X_sample, y_sample)

            self.estimators.append(model_i)

            X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(sample_rows_i)]

            prediction_oob = model_i.predict_proba(X_oob_i)

            oob_df.loc[X_oob_i.index.values, 'value'] += prediction_oob
            oob_df.loc[X_oob_i.index.values, 'count'] += 1

        self.post_fit(oob_df, y_index_sorted)

    def fit_parallel(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit()

        num_cores = multiprocessing.cpu_count()
        data = (X, y)
        oob_df = self._fit_trees_in_parallel(data, num_cores)

        y_index_sorted = y.copy()
        y_index_sorted.index = range(len(y))

        self.post_fit(oob_df, y_index_sorted)

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

        sample_rows_i = random.choices(range(len(X.index.values.tolist())),
                                       k=round(X.shape[0] * self.max_samples))

        model_i = copy.deepcopy(self.estimator)

        X_sample = X.iloc[sample_rows_i]
        y_sample = y.iloc[sample_rows_i]

        model_i.fit(X_sample, y_sample)

        self.estimators.append(model_i)

        X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(sample_rows_i)]
        prediction_oob = model_i.predict_proba(X_oob_i)

        return model_i, X_oob_i.index.values, prediction_oob

    def _combine_parameters(self, data, models):
        X, y = data
        models, oob_indices_values, predictions_oob = [[model[i] for model in models] for i in range(3)]
        self.estimators = models

        oob_df = pd.DataFrame(0.0, index=range(len(X)), columns=['value', 'count'])

        for index_values_i, prediction_oob_i in zip(oob_indices_values, predictions_oob):
            oob_df.loc[index_values_i, 'value'] += prediction_oob_i
            oob_df.loc[index_values_i, 'count'] += 1

        return oob_df

    def pre_fit(self):
        if self.random_state is not None:
            random.seed(self.random_state)
        self.estimators.clear()

    def post_fit(self, oob_df: pd.DataFrame, y_index_sorted):
        oob_prediction_mean = oob_df['value'] / oob_df['count']
        oob_prediction_mean = oob_prediction_mean.dropna()
        y_index_sorted = y_index_sorted[oob_prediction_mean.index]
        self.oob_score_ = self.__get_metric_value(oob_prediction_mean, y_index_sorted)

    @staticmethod
    def calculate_roc_auc_column_helper(row, scores):
        score_i = row.iloc[0]
        ground_truth_i = row.iloc[1]

        if ground_truth_i == 1:
            return 0

        num_positive_scores_bigger = scores[(scores['probability'] > score_i) & (scores['truth'] == 1)].shape[0]
        num_positive_scores_equal = scores[(scores['probability'] == score_i) &
                                           (scores['truth'] == 1)].shape[0]
        return num_positive_scores_bigger + num_positive_scores_equal / 2

    def __get_metric_value(self, y_predicted, y_original):
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
