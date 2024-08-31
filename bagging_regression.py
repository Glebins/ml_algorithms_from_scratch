import numpy as np
import pandas as pd
import random
import copy
import multiprocessing


def add_column_of_ones(X: pd.DataFrame):
    X1 = X.copy()
    X1.insert(0, -1, 1)
    X1.columns = pd.RangeIndex(0, len(X1.columns.values), 1)
    return X1


def generate_weights(size):
    return pd.Series([1] * size)  # naive implementation


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=1.0,
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


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric_type = metric
        self.weight_type = weight

        self.train_X = None
        self.train_y = None
        self.train_size = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train_X = X
        self.train_y = y
        self.train_size = X.shape

    def compute_distance(self, test_X: pd.DataFrame):
        assert self.train_X.shape[1] == test_X.shape[1], (f"Train and test arrays have different dimensions:"
                                                          f"{self.train_X.shape} for trains vs {test_X.shape} for tests")

        if self.metric_type == 'euclidean':
            dists = np.sqrt(np.sum(np.square(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2))
        elif self.metric_type == 'manhattan':
            dists = np.sum(np.abs(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2)
        elif self.metric_type == 'chebyshev':
            dists = np.max(np.abs(test_X.to_numpy()[:, None, :] - self.train_X.to_numpy()), axis=2)
        elif self.metric_type == 'cosine':
            dists = np.sum(test_X.to_numpy()[:, None, :] * self.train_X.to_numpy(), axis=2)
            norm1 = np.linalg.norm(test_X.to_numpy(), ord=2, axis=1)
            norm2 = np.linalg.norm(self.train_X.to_numpy(), ord=2, axis=1)
            for i in range(dists.shape[0]):
                for j in range(dists.shape[1]):
                    dists[i][j] /= (norm1[i] * norm2[j])
            dists = 1 - dists
        else:
            raise ValueError("Unknown metric")

        return dists

    def predict(self, test_X: pd.DataFrame):
        dists = self.compute_distance(test_X)
        num_test = test_X.shape[0]

        predictions = np.zeros(num_test, dtype='float')

        for i in range(num_test):
            distances_to_neighbors = dists[i]
            sorted_indices = distances_to_neighbors.argsort()
            distances_to_neighbors = distances_to_neighbors[sorted_indices][:self.k]

            neighbors_values = self.train_y.to_numpy()
            neighbors_values = neighbors_values[sorted_indices][:self.k]

            # print(distances_to_neighbors, neighbors_values)
            #
            # distances = dict(zip(dists[i], self.train_y))
            # distances = dict(sorted(distances.items()))
            # distances_k = dict(itertools.islice(distances.items(), self.k))

            # print(distances_k)

            # predictions[i] = self.get_weighted_verdict(distances_k)
            predictions[i] = self.get_weighted_verdict(distances_to_neighbors, neighbors_values)
            # print(qw)

        return np.array(predictions)

    def get_weighted_verdict(self, distances, values):
        weights = self.__get_weights(distances)
        values_of_neighbors = list(values)

        return np.dot(weights, values_of_neighbors)

    def __get_weights(self, dists_to_neighbors):
        weights = []

        if self.weight_type == 'rank':
            denominator = sum([1 / i for i in range(1, self.k + 1)])

            for i in range(1, self.k + 1):
                numerator = 1 / i
                weight = numerator / denominator
                weights.append(weight)

        elif self.weight_type == 'distance':
            denominator = sum([1 / i for i in dists_to_neighbors])

            for i in range(self.k):
                numerator = 1 / dists_to_neighbors[i]
                weight = numerator / denominator
                weights.append(weight)

        else:
            weights = [1 / self.k] * self.k

        return weights

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
            res_str += f"{key} = {value}, "

        res_str = res_str[:-2]
        res_str += ")"

        return res_str


class NotMyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = 0
        self.train_x = None
        self.train_y = None

    def __repr__(self):
        msg = "MyKNNClf class: k={}"
        return msg.format(self.k)

    def fit(self, X, y):
        self.train_x, self.train_y = self._asarray(X), self._asarray(y)
        self.train_size = X.shape

    def predict(self, X):
        X = self._asarray(X)
        distance = self._pairwise_distance(X, self.train_x)
        neighbors_target = self.train_y[np.argsort(distance)][:, : self.k]
        neighbors_distance = np.sort(distance)[:, : self.k]

        # print(neighbors_distance)

        if self.weight == "uniform":
            q = np.mean(neighbors_target, axis = 1)
            # print(q)
            return q

        elif self.weight == "rank":
            ranks = 1 / np.arange(1, self.k + 1)
            weights = ranks / np.sum(ranks)
            return np.sum(neighbors_target * weights, axis = 1)

        elif self.weight == "distance":
            total_distance = np.sum(1 / neighbors_distance, axis = 1)[:, np.newaxis]
            weights = (1 / neighbors_distance) / total_distance
            return np.sum(neighbors_target * weights, axis = 1)

    def _pairwise_distance(self, x, y):
        if self.metric == "euclidean":
            return np.sqrt(np.sum(np.abs(x[:, np.newaxis] - y) ** 2, axis = 2))

        elif self.metric == "chebyshev":
            return np.max(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "manhattan":
            return np.sum(np.abs(x[:, np.newaxis] - y), axis=2)

        elif self.metric == "cosine":
            x_norm = np.linalg.norm(x, axis=1)
            y_norm = np.linalg.norm(y, axis=1)
            similarity = np.dot(x, y.T) / (x_norm[:, np.newaxis] * y_norm)
            return 1 - similarity

        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def _asarray(self, obj):
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj)
        return obj


class Tree:
    def __init__(self, col_name=None, split_value=None, is_leaf=False):
        self.left = None
        self.right = None

        self.__col_name = col_name
        self.__split_value = split_value
        self.__depth = 0

        self.is_leaf = is_leaf
        self.leaf_value = None

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
                result += f"{(self.__depth + 1) * indent}leaf_left = {self.left.leaf_value}\n"
            else:
                result += str(self.left)

            if self.right.is_leaf:
                result += f"{(self.__depth + 1) * indent}leaf_right = {self.right.leaf_value}\n"
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


def construct_X(X):
    return pd.DataFrame(data=np.array(X))


def construct_y(y):
    return pd.Series(y)


def get_MSE_of_col(column: pd.Series):
    return column.var(ddof=0)


def get_best_split(X: pd.DataFrame, y: pd.Series, delimiters, bins, recalc_every_step):
    best_params = {'col_name': None, 'split_value': None, 'ig': 0}

    criterion_function = get_MSE_of_col

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
                #print(S_0, S_1, S_2, information_gain)
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


class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, recalc_every_step=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.recalc_every_step = recalc_every_step

        # todo delete after the course
        if self.bins is None:
            self.recalc_every_step = True

        self.remaining_leaves = 2
        self.leafs_cnt = 0
        self.tree = Tree()
        self.fi = {}

        self.__delimiters = None

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
        col_name, split_value, gain_metric = get_best_split(X, y, self.__delimiters, self.bins, self.recalc_every_step)

        #print("----", gain_metric, col_name, "-------------------------", sep="\n")

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
        #print("-----", y, "--------------\n", sep="\n")
        node.leaf_value = y.mean()

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
        for col_i in X:
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

    def predict(self, X: pd.DataFrame):
        predictions = []

        for row in X.iterrows():
            node = self.tree
            while not node.is_leaf:
                if row[1][node.col_name] > node.split_value:
                    node = node.right
                else:
                    node = node.left

            predictions.append(node.leaf_value)

        return predictions

    def print_tree(self):
        print(self.tree)

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

    def get_sum_leaves_values(self, node=None):
        result = 0
        node = self.tree if node is None else node

        if not node.is_leaf:
            result += self.get_sum_leaves_values(node.left)
            result += self.get_sum_leaves_values(node.right)
        else:
            return node.leaf_value

        return result

    def get_debug_info(self):
        return self.leafs_cnt, round(self.get_sum_leaves_values(), 6)


class NotMyTreeReg:
    def __init__(
            self,
            max_depth=5,
            min_samples_split=2,
            max_leafs=20,
            bins=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(max_leafs, 2)
        self.leafs_cnt = 0
        self.tree = {}
        self.leafs_sum = 0
        self.bins = bins
        self.splits_by_bins = {}
        self.fi = {}

    def __str__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def get_best_split(self, X, y):
        root_mse = NotMyTreeReg.calc_mse(y)
        results = {col: self.get_one_col_split(y, X[col], root_mse) for col in X}
        col_name, (split_value, gain) = max(
            filter(lambda x: x[1] is not None, results.items()),
            default=None,
            key=lambda x: x[1][1],
        )
        return col_name, split_value, gain

    @staticmethod
    def calc_mse(y):
        y_avg = np.mean(y)
        return 1 / len(y) * np.sum((y - y_avg) ** 2)

    def get_one_col_split(self, y, col, root_mse):
        if col.name in self.splits_by_bins:
            splits = dict.fromkeys(self.splits_by_bins[col.name])
        else:
            values = np.sort(col.unique())
            splits = dict.fromkeys((values[1:] + np.roll(values, 1)[1:]) / 2)
        for split in splits:
            if max(col) > split >= min(col):
                splits[split] = NotMyTreeReg.calc_gain(y, col, root_mse, split)
        return max(
            filter(lambda x: x[1] is not None, splits.items()),
            default=None,
            key=lambda x: x[1],
        )

    @staticmethod
    def calc_gain(y, col, root_mse, split):
        left = y[col <= split]
        right = y[col > split]
        left_mse = NotMyTreeReg.calc_mse(left)
        right_mse = NotMyTreeReg.calc_mse(right)
        return (
                root_mse - left_mse * len(left) / len(y) - right_mse * len(right) / len(y)
        )

    def fit(self, X, y, depth="1", n=0):
        if depth == "1":
            self.check_splits_or_bins(X)
            self.fi = dict.fromkeys(X.columns, 0)
            n = len(y)
        cols_to_split = self.check_cols_to_split(X)
        if (
                len(y) >= self.min_samples_split
                and self.max_leafs >= self.leafs_cnt + depth.count("1") + 1
                and len(depth) <= self.max_depth
                and not (bool(cols_to_split) - bool(self.bins))
        ):
            col_name, split_value, ig = self.get_best_split(X, y)

            #print("----", ig, col_name, "-------------------------", sep="\n")

            X_1 = X[X[col_name] <= split_value]
            X_2 = X[X[col_name] > split_value]
            y_1 = y[X[col_name] <= split_value]
            y_2 = y[X[col_name] > split_value]
            self.fi[col_name] += self.calc_fi(y, y_1, y_2, n)
            self.tree = {
                depth: (
                    col_name,
                    float(split_value),
                    self.fit(X_1, y_1, depth + "1", n),
                    self.fit(X_2, y_2, depth + "2", n),
                )
            }
            return self.tree
        else:
            self.leafs_cnt += 1
            #print("-----", y, "--------------\n", sep="\n")
            value = y.mean()
            self.leafs_sum += value
            return depth, "left" if depth.endswith("1") else "right", value

    def print_tree(self, tree=None):
        tab = "  "
        if tree is None:
            tree = self.tree
        for k, v in tree.items():
            print(f"{tab * (len(k) - 1)}{v[0]} > {v[1]}")
            if isinstance(v[2], dict):
                self.print_tree(v[2])
            else:
                print(f"{tab * (len(v[2][0]) - 1)}{v[2][1]} = {v[2][2]}")
            if isinstance(v[3], dict):
                self.print_tree(v[3])
            else:
                print(f"{tab * (len(v[3][0]) - 1)}{v[3][1]} = {v[3][2]}")

    def predict(self, X):
        y_pred = X.apply(self.predict_row, axis=1)
        return y_pred

    def predict_row(self, row, tree=None):
        if tree is None:
            tree = self.tree
        for _, v in tree.items():
            tree = v[3] if row[v[0]] > v[1] else v[2]
            if isinstance(tree, dict):
                return self.predict_row(row, tree)
            else:
                return tree[2]

    def check_splits_or_bins(self, X):
        for col in X:
            values = np.sort(X[col].unique())
            if self.bins and len(values) > self.bins:
                self.splits_by_bins[col] = sorted(
                    np.histogram(X[col], bins=self.bins)[1][1:-1]
                )

    def check_cols_to_split(self, X):
        cols = [
            col
            for col in X
            if col in self.splits_by_bins
               and any(max(X[col]) > i > min(X[col]) for i in self.splits_by_bins[col])
        ]
        return cols

    def calc_fi(self, y, y_1, y_2, n):
        s0, s1, s2 = (NotMyTreeReg.calc_mse(i) for i in (y, y_1, y_2))
        fi = len(y) / n * (s0 - s1 * len(y_1) / len(y) - s2 * len(y_2) / len(y))
        return fi






class MyBaggingReg:
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

    def predict(self, X: pd.DataFrame):
        X_index_sorted = X.copy()
        X_index_sorted.index = range(len(X))

        result = pd.Series([0] * len(X_index_sorted))

        for model in self.estimators:
            result += model.predict(X_index_sorted)

        result.index = X.index

        return result / self.n_estimators

    def fit_parallel(self, X: pd.DataFrame, y: pd.Series):
        self.pre_fit()

        num_cores = multiprocessing.cpu_count()
        data = (X, y)
        oob_df = self._fit_trees_in_parallel(data, num_cores)

        y_index_sorted = y.copy()
        y_index_sorted.index = range(len(y))

        self.post_fit(oob_df, y_index_sorted)

    def fit_sequential(self, X: pd.DataFrame, y: pd.Series):
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

        oob_df = pd.DataFrame(0, index=X_index_sorted.index, columns=['value', 'count'])

        for i in range(self.n_estimators):
            sample_rows_i = sample_rows[i]
            model_i = copy.deepcopy(self.estimator)

            X_sample = X.iloc[sample_rows_i]
            y_sample = y.iloc[sample_rows_i]

            model_i.fit(X_sample, y_sample)

            self.estimators.append(model_i)

            X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(sample_rows_i)]
            prediction_oob = model_i.predict(X_oob_i)

            oob_df.loc[X_oob_i.index.values, 'value'] += prediction_oob
            oob_df.loc[X_oob_i.index.values, 'count'] += 1

        self.post_fit(oob_df, y_index_sorted)

    def pre_fit(self):
        if self.random_state is not None:
            random.seed(self.random_state)
        self.estimators.clear()

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

        sample_rows_i = random.choices(range(len(X.index.values.tolist())),
                                           k=round(X.shape[0] * self.max_samples))

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

        X_oob_i = X_index_sorted.iloc[~X_index_sorted.index.isin(sample_rows_i)]
        prediction_oob = model_i.predict(X_oob_i)

        return model_i, X_oob_i.index.values, prediction_oob

    def _combine_parameters(self, data, models):
        X, y = data
        models, oob_indices_values, predictions_oob = [[model[i] for model in models] for i in range(3)]
        self.estimators = models

        oob_df = pd.DataFrame(0, index=range(len(X)), columns=['value', 'count'])

        for index_values_i, prediction_oob_i in zip(oob_indices_values, predictions_oob):
            oob_df.loc[index_values_i, 'value'] += prediction_oob_i
            oob_df.loc[index_values_i, 'count'] += 1

        return oob_df

    def __get_metric_value(self, y_hat, y):
        if self.oob_metric_type == 'mae':
            metric_value = (y_hat - y).abs().sum() / len(y)
        elif self.oob_metric_type == 'mse':
            metric_value = np.square(y_hat - y).sum() / len(y)
        elif self.oob_metric_type == 'rmse':
            metric_value = np.sqrt(np.square(y_hat - y).sum() / len(y))
        elif self.oob_metric_type == 'mape':
            metric_value = 100 / len(y) * ((y_hat - y) / y).abs().sum()
        elif self.oob_metric_type == 'r2':
            metric_value = 1 - (np.square(y - y_hat)).sum() / (np.square(y - y.mean())).sum()
        elif self.oob_metric_type is None:
            metric_value = None
        else:
            raise ValueError('Unknown metric type')

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
