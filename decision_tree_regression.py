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
        self.__elements_in_leaves = []

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
        self.__elements_in_leaves.clear()

    def post_fit(self, X: pd.DataFrame):
        for col_i in X:
            self.fi[col_i] /= len(X)

    def fitting_process(self, X: pd.DataFrame, y: pd.Series, node, do_write_elements_in_leaves):
        processing_result = self.handle_node(X, y, node)
        if processing_result is None:
            if do_write_elements_in_leaves:
                self.__elements_in_leaves.append(y.index.values)
            return

        X_left, y_left, X_right, y_right = processing_result

        self.remaining_leaves += 1
        self.fitting_process(X_left, y_left, node.left, do_write_elements_in_leaves)
        self.remaining_leaves -= 1
        self.fitting_process(X_right, y_right, node.right, do_write_elements_in_leaves)

    def fit(self, X: pd.DataFrame, y: pd.Series, do_write_elements_in_leaves=False):
        node_start = self.tree

        self.pre_fit(X)
        self.fitting_process(X, y, node_start, do_write_elements_in_leaves)
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

    def get_elements_in_leaves(self):
        return self.__elements_in_leaves

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
