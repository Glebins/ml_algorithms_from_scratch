import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = []
        self.inertia_ = 0

    def fit(self, X: pd.DataFrame):
        if self.random_state:
            np.random.seed(self.random_state)

        best_centers = []
        best_inertia = float('inf')

        for i in range(self.n_init):
            self.fit_one_time(X)

            # print(self.inertia_)

            if self.inertia_ < best_inertia:
                best_centers = self.cluster_centers_
                best_inertia = self.inertia_

        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia

    def fit_one_time(self, X: pd.DataFrame):
        self.cluster_centers_ = []

        for i in range(self.n_clusters):
            centroid_i = [np.random.uniform(X[i].min(), X[i].max()) for i in X]
            self.cluster_centers_.append(centroid_i)

        for i in range(self.max_iter):
            distances = self.find_distance(X.to_numpy())
            points_by_clusters = np.argmin(distances, axis=-1)

            new_cluster_centers_ = []
            for j in range(self.n_clusters):
                index_of_points_in_cluster = np.where(points_by_clusters == j)[0]

                if len(index_of_points_in_cluster) == 0:
                    new_cluster_centers_.append(self.cluster_centers_[j])
                    continue

                X_points_in_cluster = X.iloc[index_of_points_in_cluster, :]

                cluster_i_new_center = []
                for col_i in X_points_in_cluster:
                    cluster_i_new_center.append(X_points_in_cluster[col_i].mean())
                new_cluster_centers_.append(cluster_i_new_center)

            # print(new_cluster_centers_)
            # print(self.cluster_centers_)
            # print(self.calculate_WCSS(X, points_by_clusters))
            # print("________________________")

            if np.array_equal(np.array(self.cluster_centers_).round(3), np.array(new_cluster_centers_).round(3)):
                self.inertia_ = self.calculate_WCSS(X, points_by_clusters)
                break

            self.cluster_centers_ = new_cluster_centers_

    def predict(self, X: pd.DataFrame):
        distances = self.find_distance(X.to_numpy())
        points_by_clusters = np.argmin(distances, axis=-1)
        return points_by_clusters

    def calculate_WCSS(self, X: pd.DataFrame, points_by_clusters):
        distance = 0
        for i in range(self.n_clusters):
            index_of_points_in_cluster = np.where(points_by_clusters == i)[0]
            X_points_in_cluster = X.iloc[index_of_points_in_cluster, :]
            distance += (self.find_distance(X_points_in_cluster.to_numpy(), self.cluster_centers_[i]) ** 2).sum()

        return distance

    def find_distance(self, X: np.array, second_array=None):
        if second_array is None:
            second_array = self.cluster_centers_
        return np.sqrt(((X[:, None, :] - np.array(second_array)) ** 2).sum(axis=-1))

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
