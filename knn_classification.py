import pandas as pd
import numpy as np


class KNNClf:
    def __init__(self, k=3, metric_type='euclidean', weight_type='uniform'):
        self.k = k
        self.metric_type = metric_type
        self.weight_type = weight_type

        self.train_X = None
        self.train_y = None

    def fit(self, train_X: pd.DataFrame, train_y: pd.Series):
        self.train_X = train_X.copy()
        self.train_y = train_y.copy()

    def compute_distance(self, test_X: pd.DataFrame):
        assert self.train_X.shape[1] == test_X.shape[1], (f"Train and test arrays have different dimensions:"
                                                          f"{self.train_X.shape} for trains vs {test_X.shape} for tests")

        dists = np.zeros((test_X.shape[0], self.train_X.shape[0]))

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
        return dists

    def get_weighted_verdict(self, distances, values):
        classes = list(set(values))
        ranks = {}
        dists = {}

        i = 1
        for dist, cls in zip(distances, values):
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
            denominator = sum([1 / i for i in range(1, len(values) + 1)])
            probabilities_by_rank = {}

            for cls in classes:
                numerator = sum([1 / i for i in ranks[cls]])
                probabilities_by_rank[cls] = numerator / denominator

            return probabilities_by_rank

        elif self.weight_type == 'distance':
            denominator = sum([1 / i for i in distances])
            probabilities_by_distance = {}

            for cls in classes:
                numerator = sum([1 / i for i in dists[cls]])
                probabilities_by_distance[cls] = numerator / denominator

            return probabilities_by_distance

        else:
            nearest_neighbors = np.array(list(values))
            probabilities_by_quantity = {}

            for cls in classes:
                probabilities_by_quantity[cls] = (nearest_neighbors == cls).mean()

            return probabilities_by_quantity

    def predict(self, test_X):
        dists = self.compute_distance(test_X)
        num_test = test_X.shape[0]

        predictions = np.zeros(num_test, dtype='int')

        for i in range(num_test):

            distances_to_neighbors = dists[i]
            sorted_indices = distances_to_neighbors.argsort()
            distances_to_neighbors = distances_to_neighbors[sorted_indices][:self.k]

            neighbors_values = self.train_y.to_numpy()
            neighbors_values = neighbors_values[sorted_indices][:self.k]

            probabilities_by_class = self.get_weighted_verdict(distances_to_neighbors, neighbors_values)
            probabilities_by_class = dict(sorted(probabilities_by_class.items(), key=lambda item: -item[1]))

        return np.array(predictions)

    def predict_proba(self, test_X):
        dists = self.compute_distance(test_X)
        num_test = test_X.shape[0]

        predictions = np.zeros(num_test, dtype='float')

        for i in range(num_test):

            distances_to_neighbors = dists[i]
            sorted_indices = distances_to_neighbors.argsort()
            distances_to_neighbors = distances_to_neighbors[sorted_indices][:self.k]

            neighbors_values = self.train_y.to_numpy()
            neighbors_values = neighbors_values[sorted_indices][:self.k]

            probabilities_by_class = self.get_weighted_verdict(distances_to_neighbors, neighbors_values)
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
