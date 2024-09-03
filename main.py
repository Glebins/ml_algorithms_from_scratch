import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
import math
import time

from sklearn import *

from gradient_boosting_regression import *

matplotlib.use('TkAgg')


def main():
    X, y = datasets.make_classification(n_samples=170, n_features=10, n_informative=7, random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=20, random_state=42)

    boost_reg = MyBoostReg()
    boost_reg.fit(X_train, y_train)

    print(boost_reg)


if __name__ == '__main__':
    main()
