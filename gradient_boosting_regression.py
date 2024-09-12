import random

from decision_tree_regression import *


class MyBoostReg:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_features=0.5, max_samples=0.5, random_state=42,
                 reg=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, loss='MSE', metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_type = loss
        self.metric_type = metric
        self.reg_coef = reg

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.__test_loss_and_metric_types()

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = None
        self.trees = []
        self.best_score = 0
        self.fi = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        self.pred_0 = self.calculate_best_prediction(y)
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
        prev_score_valid = float('inf')
        last_best_score = 0

        if doing_early_stopping:
            X_eval.index = range(X_eval.shape[0])
            y_eval.index = range(len(y_eval))

        current_pred = pd.Series([self.pred_0] * len(y))
        eval_predict = pd.Series([self.pred_0] * len(y_eval)) if doing_early_stopping else 0
        sum_prev_predictions = current_pred

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

            grad = self.calculate_gradient(sum_prev_predictions, y)
            grad_part = grad.iloc[rows_idx]
            grad_part.index = range(grad_part.shape[0])

            tree_i.fit(X_train_i, -grad_part, do_write_elements_in_leaves=True)

            remainder = y - sum_prev_predictions
            remainder_part = remainder.iloc[rows_idx]
            remainder_part.index = range(remainder_part.shape[0])

            remainder_part += self.reg_coef * current_amount_of_leaves
            current_amount_of_leaves += tree_i.leafs_cnt

            self.recalculate_leaves_values(tree_i.tree, tree_i.get_elements_in_leaves(), remainder_part)

            current_pred = pd.Series(tree_i.predict(X))

            if callable(self.learning_rate):
                sum_prev_predictions += self.learning_rate(i) * current_pred
            else:
                sum_prev_predictions += self.learning_rate * current_pred

            self.trees.append(tree_i)

            for col_i in self.fi.keys():
                self.fi[col_i] += (tree_i.fi[col_i] if col_i in tree_i.fi else 0) * len(X_train_i) / len(X)

            loss_value = self.calculate_metric(sum_prev_predictions, y, self.loss_type)

            metric_value = None
            if self.metric_type is not None:
                metric_value = self.calculate_metric(sum_prev_predictions, y, self.metric_type)

            validation_value = None
            validation_metric = None
            if doing_early_stopping:
                current_eval_pred = pd.Series(tree_i.predict(X_eval))

                if callable(self.learning_rate):
                    eval_predict += self.learning_rate(i) * current_eval_pred
                else:
                    eval_predict += self.learning_rate * current_eval_pred

                validation_metric = self.metric_type if self.metric_type is not None else self.loss_type
                validation_value = self.calculate_metric(eval_predict, y_eval, validation_metric)

                if self.__is_non_improvement(prev_score_valid, validation_value, validation_metric):
                    steps_of_non_improvement_valid += 1
                else:
                    last_best_score = validation_value
                    steps_of_non_improvement_valid = 0
                prev_score_valid = validation_value

                if steps_of_non_improvement_valid == early_stopping:
                    if verbose is not None:
                        print(f"---------------Validation = {validation_value}, early stopping---------------")
                    del self.trees[-early_stopping:]
                    self.best_score = last_best_score
                    break

            if verbose is not None and (i % verbose == 0 or i == self.n_estimators):
                self.__debug_while_fit(i, loss_value, metric_value, validation_metric, validation_value)

        else:
            if self.metric_type is not None:
                self.best_score = self.calculate_metric(sum_prev_predictions, y, self.metric_type)
            else:
                self.best_score = self.calculate_metric(sum_prev_predictions, y, self.loss_type)

        X.index = train_index
        y.index = train_index
        if validation_index is not None:
            X_eval.index = validation_index
            y_eval.index = validation_index

    def predict(self, X: pd.DataFrame):
        result = pd.Series([self.pred_0] * X.shape[0])

        for i, tree_i in enumerate(self.trees):
            if callable(self.learning_rate):
                result += self.learning_rate(i + 1) * pd.Series(tree_i.predict(X))
            else:
                result += self.learning_rate * pd.Series(tree_i.predict(X))

        result.index = X.index

        return result

    def calculate_best_prediction(self, y):
        if self.loss_type == 'MSE':
            return y.mean()
        elif self.loss_type == 'MAE':
            return y.median()
        else:
            raise ValueError("Unknown loss type")

    def calculate_gradient(self, y_pred, y) -> pd.Series:
        if self.loss_type == 'MSE':
            return 2 * (y_pred - y)
        elif self.loss_type == 'MAE':
            return pd.Series(np.sign(y_pred - y))
        else:
            raise ValueError("Unknown loss type")

    @staticmethod
    def calculate_metric(y_pred, y, metric_type):
        if metric_type == 'MSE':
            return ((y_pred - y) ** 2).mean()
        elif metric_type == 'MAE':
            return (y_pred - y).abs().mean()
        elif metric_type == 'RMSE':
            return np.sqrt(((y_pred - y) ** 2).mean())
        elif metric_type == 'R2':
            return 1 - np.square(y - y_pred).sum() / np.square(y - y.mean()).sum()
        elif metric_type == 'MAPE':
            return 100 / len(y) * ((y_pred - y) / y).abs().sum()
        else:
            raise ValueError("Unknown metric type")

    def recalculate_leaves_values(self, tree, indices_elements_in_leaves, y, i=0):
        if tree.depth == 0:
            i = 0

        if not tree.is_leaf:
            i = self.recalculate_leaves_values(tree.left, indices_elements_in_leaves, y, i)
            i = self.recalculate_leaves_values(tree.right, indices_elements_in_leaves, y, i)
        else:
            tree.leaf_value = self.calculate_best_prediction(y.iloc[indices_elements_in_leaves[i]])
            i += 1

        if tree.depth != 0:
            return i

    def __debug_while_fit(self, i, loss_value, metric_value, validation_metric, validation_value):
        metric_part = ""
        validation_part = ""
        if metric_value is not None:
            metric_part = f",\tMetric({self.metric_type}) = {metric_value}"
        if validation_value is not None:
            validation_part = f",\tValidation({validation_metric}) = {validation_value}"
        print(f"{i}. Loss({self.loss_type}) = {loss_value}" + metric_part + validation_part)

    def __test_loss_and_metric_types(self):
        loss_types = ('MSE', 'MAE')
        metric_types = ('MSE', 'MAE', 'RMSE', 'R2', 'MAPE')

        if self.loss_type not in loss_types:
            raise ValueError(f"Loss type must be one of this: {loss_types}")
        if self.metric_type not in metric_types and self.metric_type is not None:
            raise ValueError(f"Metric type must be one of this: {metric_types}")

    @staticmethod
    def __test_early_stopping(X_eval, y_eval, early_stopping):
        if X_eval is not None and y_eval is not None and early_stopping is not None:
            return True
        else:
            return False

    @staticmethod
    def __is_non_improvement(prev_val, current_val, metric):
        if metric in ('MSE', 'RMSE', 'MAE', 'MAPE'):
            return current_val >= prev_val
        else:
            return current_val <= prev_val

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
