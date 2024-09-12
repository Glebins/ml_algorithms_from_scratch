import numpy as np
import pandas as pd


class MyPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit_transform(self, X: pd.DataFrame):
        X_scaled = X - X.mean(axis=0)
        covariance_matrix = X_scaled.cov().to_numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, sorted_indices][:, :self.n_components]

        return X_scaled @ eigenvectors

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
