import numpy as np
import pandas as pd
import random
import copy
import multiprocessing


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


import numpy as np
import pandas as pd


class Tree:
    def __init__(self, col_name=None, split_value=None, is_leaf=False):
        self.left = None
        self.right = None

        self.__col_name = col_name
        self.__split_value = split_value
        self.__depth = 0

        self.is_leaf = is_leaf
        self.first_class_prob = None

    def add_node(self):
        self.left = Tree()
        self.right = Tree()
        self.left.__depth = self.__depth + 1
        self.right.__depth = self.__depth + 1

    def __str__(self):
        result = ""
        indent = ' ' * 2

        if not self.is_leaf:
            result += f"{self.__depth * indent}{self.col_name} > {self.split_value}\n"

            if self.left.is_leaf:
                result += f"{(self.__depth + 1) * indent}leaf_left = {self.left.first_class_prob}\n"
            else:
                result += str(self.left)

            if self.right.is_leaf:
                result += f"{(self.__depth + 1) * indent}leaf_right = {self.right.first_class_prob}\n"
            else:
                result += str(self.right)

        return result

    @property
    def col_name(self):
        return self.__col_name

    @col_name.setter
    def col_name(self, col_name_new):
        self.__col_name = col_name_new

    @property
    def split_value(self):
        return self.__split_value

    @split_value.setter
    def split_value(self, split_value_new):
        self.__split_value = split_value_new

    @property
    def depth(self):
        return self.__depth


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


def get_gini_coefficient_of_col(column: pd.Series):
    len_objects = len(column)
    unique_objects = column.value_counts()

    gini = 1
    for item, item_repetitions in unique_objects.items():
        p_i = item_repetitions / len_objects
        gini -= p_i ** 2

    return gini


def get_best_split(X: pd.DataFrame, y: pd.Series, delimiters, criterion, bins, recalc_every_step):
    criterion_function = get_entropy_of_col if criterion == 'entropy' else get_gini_coefficient_of_col

    best_params = {'col_name': None, 'split_value': None, 'ig': 0}

    S_0 = criterion_function(y)

    for column_i in X:
        # unique_values = sorted(X[column_i].unique())
        # delimiters = []
        # for i in range(len(unique_values) - 1):
        #     mean_i = (unique_values[i] + unique_values[i + 1]) / 2
        #     delimiters.append(mean_i)

        if recalc_every_step:
            delimiters = split_into_delimiters(X, bins)

        len_objects = len(X[column_i])

        delimiters_i = delimiters[column_i]

        for delim in delimiters_i:
            X_left = X[X[column_i] <= delim][column_i]
            X_right = X[X[column_i] > delim][column_i]

            y_left = y[X[column_i] <= delim]
            y_right = y[X[column_i] > delim]

            if X_left.empty or X_right.empty:
                continue

            S_1 = criterion_function(y_left)
            S_2 = criterion_function(y_right)

            information_gain = S_0 - len(y_left) / len_objects * S_1 - len(y_right) / len_objects * S_2

            if information_gain > best_params['ig']:
                best_params['ig'] = information_gain
                best_params['col_name'] = column_i
                best_params['split_value'] = delim

    return [best_params['col_name'], best_params['split_value'], best_params['ig']]


def split_into_delimiters(X: pd.DataFrame, bins):
    delimiters = {}

    for column_i in X:
        unique_values = sorted(X[column_i].unique())

        if bins is not None and bins < len(unique_values):
            delimiters_i = split_using_histogram(unique_values, bins)

        else:
            delimiters_i = split_using_own_values(unique_values)
        delimiters[column_i] = delimiters_i

    return delimiters


def split_using_histogram(unique_values, bins):
    _, delimiters_i = np.histogram(unique_values, bins=bins)
    delimiters_i = delimiters_i[1:-1]

    return delimiters_i


def split_using_own_values(unique_values):
    delimiters_i = []
    for i in range(len(unique_values) - 1):
        mean_i = (unique_values[i] + unique_values[i + 1]) / 2
        delimiters_i.append(mean_i)
    return delimiters_i


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy',
                 recalc_every_step=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.recalc_every_step = recalc_every_step
        self.__test_criterion()

        # todo delete after the course
        if self.bins is None:
            self.recalc_every_step = True

        self.remaining_leaves = 2
        self.leafs_cnt = 0
        self.tree = Tree()
        self.fi = {}

        self.__delimiters = None

    def __test_criterion(self):
        if self.criterion not in ('entropy', 'gini'):
            raise ValueError("Criterion must be either 'entropy' or 'gini'")

    def is_node_a_leaf(self, X: pd.DataFrame, y: pd.Series, node: Tree):
        if node.depth == 0:
            return False
        if len(y.unique()) == 1:
            return True
        if self.remaining_leaves + self.leafs_cnt > self.max_leafs:
            return True
        if len(y) < self.min_samples_split:
            return True
        if node.depth == self.max_depth:
            return True
        return False

    def __handle_node_as_node(self, X: pd.DataFrame, y: pd.Series, node: Tree):
        col_name, split_value, gain_metric = get_best_split(X, y, self.__delimiters, self.criterion, self.bins, self.recalc_every_step)

        # make a human check (the same elements in bins)
        if col_name is None:
            return self.__handle_node_as_leaf(X, y, node)

        X_left = X[X[col_name] <= split_value]
        X_right = X[X[col_name] > split_value]

        y_left = y[X[col_name] <= split_value]
        y_right = y[X[col_name] > split_value]

        self.fi[col_name] += len(X) * gain_metric

        node.col_name = col_name
        node.split_value = split_value
        node.add_node()

        return [X_left, y_left, X_right, y_right]

    def __handle_node_as_leaf(self, X: pd.DataFrame, y: pd.Series, node: Tree):
        self.leafs_cnt += 1
        node.is_leaf = True
        node.first_class_prob = y.mean()

    def handle_node(self, X: pd.DataFrame, y: pd.Series, node: Tree):
        if self.is_node_a_leaf(X, y, node):
            self.__handle_node_as_leaf(X, y, node)
            return None
        return self.__handle_node_as_node(X, y, node)

    def pre_fit(self, X: pd.DataFrame):
        self.fi.clear()
        for col_i in X:
            self.fi[col_i] = 0

        if not self.recalc_every_step:
            self.__delimiters = split_into_delimiters(X, self.bins)

    def post_fit(self, X: pd.DataFrame):
        for col_i in self.fi:
            self.fi[col_i] /= len(X)

    def fitting_process(self, X: pd.DataFrame, y: pd.Series, node):
        processing_result = self.handle_node(X, y, node)
        if processing_result is None:
            return

        X_left, y_left, X_right, y_right = processing_result

        self.remaining_leaves += 1
        self.fitting_process(X_left, y_left, node.left)
        self.remaining_leaves -= 1
        self.fitting_process(X_right, y_right, node.right)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        node_start = self.tree

        self.pre_fit(X)
        self.fitting_process(X, y, node_start)
        self.post_fit(X)

    def print_tree(self):
        print(self.tree)

    def predict_proba(self, X: pd.DataFrame):
        predictions = []

        for row in X.iterrows():
            node = self.tree
            while not node.is_leaf:
                if row[1][node.col_name] > node.split_value:
                    node = node.right
                else:
                    node = node.left

            predictions.append(node.first_class_prob)

        return predictions

    def predict(self, X: pd.DataFrame):
        predictions = []

        for row in X.iterrows():
            node = self.tree
            while not node.is_leaf:
                if row[1][node.col_name] > node.split_value:
                    node = node.right
                else:
                    node = node.left

            predictions.append(1 if node.first_class_prob > 0.5 else 0)

        return predictions

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

    def get_sum_leaves_values(self, node=None):
        result = 0
        node = self.tree if node is None else node

        if not node.is_leaf:
            result += self.get_sum_leaves_values(node.left)
            result += self.get_sum_leaves_values(node.right)
        else:
            return node.first_class_prob

        return result

    def get_debug_info(self):
        return self.leafs_cnt, round(self.get_sum_leaves_values(), 6)


import pandas as pd
import numpy as np
import itertools


import pandas as pd
import numpy as np
import itertools
import collections


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric_name = metric
        self.weight_type = weight

        self.train_size = None
        self.train_X = None
        self.train_y = None

    def fit(self, train_X: pd.DataFrame, train_y: pd.Series):
        self.train_X = train_X.copy()
        self.train_y = train_y.copy()
        self.train_size = train_y.shape

    def compute_distance(self, test_X: pd.DataFrame):
        assert self.train_X.shape[1] == test_X.shape[1], (f"Train and test arrays have different dimensions:"
                                                          f"{self.train_X.shape} for trains vs {test_X.shape} for tests")

        dists = np.zeros((test_X.shape[0], self.train_X.shape[0]))

        if self.metric_name == 'euclidean':
            dists = np.sqrt(np.sum(np.square(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2))
        elif self.metric_name == 'manhattan':
            dists = np.sum(np.abs(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2)
        elif self.metric_name == 'chebyshev':
            dists = np.max(np.abs(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2)
        elif self.metric_name == 'cosine':
            dists = np.sum(test_X.to_numpy()[:, None, :] * self.train_X.to_numpy(), axis=2)
            norm1 = np.linalg.norm(test_X.to_numpy(), ord=2, axis=1)
            norm2 = np.linalg.norm(self.train_X.to_numpy(), ord=2, axis=1)
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    dists[i][j] /= (norm1[i] * norm2[j])
            dists = 1 - dists
        return dists

    def get_weighted_verdict(self, distances):
        classes = list(set(distances.values()))
        ranks = {}
        dists = {}

        i = 1
        for dist, cls in distances.items():
            if cls not in ranks:
                ranks[cls] = [i]
            else:
                ranks[cls].append(i)

            if cls not in dists:
                dists[cls] = [dist]
            else:
                dists[cls].append(dist)

            i += 1

        if self.weight_type == 'rank':
            denominator = sum([1 / i for i in range(1, len(distances.values()) + 1)])
            probabilities_by_rank = {}

            for cls in classes:
                numerator = sum([1 / i for i in ranks[cls]])
                probabilities_by_rank[cls] = numerator / denominator

            return probabilities_by_rank

        elif self.weight_type == 'distance':
            denominator = sum([1 / i for i in distances.keys()])
            probabilities_by_distance = {}

            for cls in classes:
                numerator = sum([1 / i for i in dists[cls]])
                probabilities_by_distance[cls] = numerator / denominator

            return probabilities_by_distance

        else:
            nearest_neighbors = np.array(list(distances.values()))
            probabilities_by_quantity = {}

            for cls in classes:
                probabilities_by_quantity[cls] = (nearest_neighbors == cls).mean()

            return probabilities_by_quantity

    def predict(self, test_X):
        dists = self.compute_distance(test_X)
        num_test = test_X.shape[0]

        predictions = np.zeros(num_test, dtype='int')

        for i in range(num_test):
            distances = dict(zip(dists[i], self.train_y))
            distances = dict(sorted(distances.items()))
            distances_k = dict(itertools.islice(distances.items(), self.k))
            probabilities_by_class = self.get_weighted_verdict(distances_k)
            probabilities_by_class = dict(sorted(probabilities_by_class.items(), key=lambda item: -item[1]))

            # todo delete after the course
            if len(probabilities_by_class.values()) == 2 and probabilities_by_class[1] == probabilities_by_class[0]:
                predictions[i] = 1
            else:
                predictions[i] = list(probabilities_by_class.keys())[0]

        return np.array(predictions)

    def predict_proba(self, test_X):
        dists = self.compute_distance(test_X)
        num_test = test_X.shape[0]

        predictions = np.zeros(num_test, dtype='float')

        for i in range(num_test):
            distances = dict(zip(dists[i], self.train_y))
            distances = dict(sorted(distances.items()))
            distances_k = dict(itertools.islice(distances.items(), self.k))
            probabilities_by_class = self.get_weighted_verdict(distances_k)
            probabilities_by_class = dict(sorted(probabilities_by_class.items(), key=lambda item: -item[1]))

            predictions[i] = 0 if 1 not in probabilities_by_class else probabilities_by_class[1]

        return np.array(predictions)

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














class MyBaggingClf:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit()

        sample_rows = []

        X_index_sorted = X.copy()
        y_index_sorted = y.copy()
        X_index_sorted.index = range(len(X))
        y_index_sorted.index = range(len(y))

        for i in range(self.n_estimators):
            sample_rows_idx_i = random.choices(range(len(X.index.values.tolist())),
                                               k=round(X.shape[0] * self.max_samples))
            sample_rows.append(sample_rows_idx_i)

        # oob_df = pd.DataFrame(0, index=X_index_sorted.index, columns=['value', 'count'])

        for i in range(self.n_estimators):
            sample_rows_i = sample_rows[i]
            model_i = copy.deepcopy(self.estimator)

            X_sample = X.iloc[sample_rows_i]
            y_sample = y.iloc[sample_rows_i]

            model_i.fit(X_sample, y_sample)

            self.estimators.append(model_i)

            # X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(sample_rows_i)]
            # prediction_oob = model_i.predict(X_oob_i)
            #
            # oob_df.loc[X_oob_i.index.values, 'value'] += prediction_oob
            # oob_df.loc[X_oob_i.index.values, 'count'] += 1

        # self.post_fit(oob_df, y_index_sorted)

    def predict_proba(self, X: pd.DataFrame):
        predictions = []

        for model in self.estimators:
            predictions.append(model.predict(X))

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

    def pre_fit(self):
        if self.random_state is not None:
            random.seed(self.random_state)
        self.estimators.clear()

    def post_fit(self, oob_df: pd.DataFrame, y_index_sorted):
        pass
        # oob_prediction_mean = oob_df['value'] / oob_df['count']
        # oob_prediction_mean = oob_prediction_mean.dropna()
        # y_index_sorted = y_index_sorted[oob_prediction_mean.index]
        # self.oob_score_ = self.__get_metric_value(oob_prediction_mean, y_index_sorted)

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
