import pandas as pd
import numpy as np


class KNNReg:
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
