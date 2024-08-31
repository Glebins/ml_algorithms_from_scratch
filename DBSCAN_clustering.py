import numpy as np
import pandas as pd


class MyDBSCAN:
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric_type = metric

        self.__points_info = []
        self.clusters_number = 0

    def fit_predict(self, X: pd.DataFrame, return_statuses_points=False):
        X_numpy = X.to_numpy()
        size = X_numpy.shape[0]
        self.__points_info = [range(X.shape[0]), [[]] * size, [-1] * size, [-1] * size, [False] * size]
        self.clusters_number = 1

        for i, elem in enumerate(self.__points_info[0]):
            self.__points_info[1][i] = self.find_neighbors_of_the_point(X_numpy, i)

        queue = list(range(X.shape[0]))

        while queue:
            elem = queue.pop(0)
            self.handle_element(elem)

        points_by_cluster = [self.clusters_number if elem == -1 else elem for elem in self.__points_info[3]]

        if (-1) not in self.__points_info[3]:
            self.clusters_number -= 1

        if return_statuses_points:
            return points_by_cluster, self.__points_info[2]
        else:
            return points_by_cluster

    def handle_element(self, elem):
        neighbors = self.__points_info[1][elem]

        if self.__points_info[2][elem] != -1 or self.__points_info[4][elem]:
            return

        if len(neighbors) < self.min_samples:
            self.__points_info[2][elem] = 1

        else:
            self.__points_info[2][elem] = 3
            self.__points_info[3][elem] = self.clusters_number
            self.__points_info[4][elem] = True

            for neighbor in neighbors:
                if self.__points_info[2][neighbor] in (-1, 1) and not self.__points_info[4][neighbor]:
                    self.handle_neighbor(neighbor)
            self.clusters_number += 1

    def handle_neighbor(self, elem):
        neighbors = self.__points_info[1][elem]

        if len(neighbors) < self.min_samples:
            self.__points_info[2][elem] = 2
            self.__points_info[3][elem] = self.clusters_number
            self.__points_info[4][elem] = True

        else:
            self.__points_info[2][elem] = 3
            self.__points_info[3][elem] = self.clusters_number
            self.__points_info[4][elem] = True

            for neighbor in neighbors:
                if self.__points_info[2][neighbor] in (-1, 1) and not self.__points_info[4][neighbor]:
                    self.handle_neighbor(neighbor)

    def find_neighbors_of_the_point(self, data, i):
        dists = self.find_distance(data, data[i])
        neighbors = np.argwhere(dists < self.eps).flatten()
        neighbors = np.delete(neighbors, np.argwhere(neighbors == i))
        return neighbors

    def find_distance(self, X: np.array, point):
        if self.distance_metric_type == 'euclidean':
            dists = np.sqrt(np.sum((X - point) ** 2, axis=-1))
        elif self.distance_metric_type == 'chebyshev':
            dists = np.max(np.abs(X - point), axis=-1)
        elif self.distance_metric_type == 'manhattan':
            dists = np.sum(np.abs(X - point), axis=-1)
        elif self.distance_metric_type == 'cosine':
            dists = np.sum(X * point, axis=-1)
            norm1 = np.linalg.norm(X, ord=2, axis=1)
            norm2 = np.linalg.norm(point, ord=2)
            for i in range(dists.shape[0]):
                dists[i] /= (norm1[i] * norm2)
            dists = 1 - dists
        else:
            raise ValueError('Unknown metric. Use one of the following: euclidean, chebyshev, manhattan, cosine')

        return dists

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
