import numpy as np
import pandas as pd


class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42):
        self.base_estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

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
