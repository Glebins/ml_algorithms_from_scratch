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
