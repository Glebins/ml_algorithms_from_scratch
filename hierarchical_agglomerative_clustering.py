import pandas as pd
import numpy as np


class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.distance_metric_type = metric

    def fit_predict(self, X: pd.DataFrame):
        size = X.shape[0]
        current_clusters_number = size

        X_index_sorted = X.copy()
        X_index_sorted.index = range(size)
        X_numpy = X_index_sorted.to_numpy()

        points_by_clusters = {i: [i] for i in range(size)}

        while current_clusters_number != self.n_clusters:
            distance_matrix, min_params = self.get_distance_matrix(X_numpy)
            i_min, j_min = min_params

            points_by_clusters = self.unite_clusters(points_by_clusters, j_min, i_min)
            current_clusters_number -= 1

            X_numpy = self.recalculate_data(X_index_sorted, points_by_clusters)

        result_clusters = []

        for i in range(size):
            for k, v in points_by_clusters.items():
                if i in v:
                    result_clusters.append(k)
                    break

        return result_clusters

    def get_distance_matrix(self, data):
        size = data.shape[0]
        distance_matrix = np.zeros((size, size))

        min_distance = float('inf')
        min_params = ()

        for i in range(size):
            for j in range(i + 1):
                point_a = data[i]
                point_b = data[j]
                distance = self.find_distance(point_a, point_b)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

                if distance < min_distance and i != j:
                    min_distance = distance
                    min_params = i, j

        return distance_matrix, min_params

    def find_distance(self, point_a, point_b):
        if self.distance_metric_type == 'euclidean':
            dists = np.sqrt(np.sum((point_a - point_b) ** 2, axis=-1))
        elif self.distance_metric_type == 'chebyshev':
            dists = np.max(np.abs(point_a - point_b), axis=-1)
        elif self.distance_metric_type == 'manhattan':
            dists = np.sum(np.abs(point_a - point_b), axis=-1)
        elif self.distance_metric_type == 'cosine':
            dists = np.sum(point_a * point_b, axis=-1)
            norm1 = np.linalg.norm(point_a, ord=2)
            norm2 = np.linalg.norm(point_b, ord=2)
            dists /= (norm1 * norm2)
            dists = 1 - dists
        else:
            raise ValueError('Unknown metric. Use one of the following: euclidean, chebyshev, manhattan, cosine')

        return dists

    @staticmethod
    def unite_clusters(points_by_clusters, cluster_a, cluster_b):
        points_by_clusters[cluster_a] += points_by_clusters[cluster_b]
        del points_by_clusters[cluster_b]
        return {k: v for k, v in enumerate(points_by_clusters.values())}

    @staticmethod
    def recalculate_data(data: pd.DataFrame, points_by_clusters: dict):
        X_result = np.zeros((len(points_by_clusters), data.shape[1]))

        for k, v in points_by_clusters.items():
            X_result[k] = data.iloc[v].mean().to_numpy()

        return X_result

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
