import pandas as pd
import numpy as np


class LogisticReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        res_str = "MyLogReg class: "
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key}={value}, "

        res_str = res_str[:-2]
        return res_str

    def __repr__(self):
        res_str = "LinearReg("

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            res_str += f"{key}={value}, "

        res_str = res_str[:-2]
        res_str += ")"

        return res_str
