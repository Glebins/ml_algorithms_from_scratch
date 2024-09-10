import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
import math
import time

from sklearn import *

from gradient_boosting_regression import *
from bagging_classification import *

matplotlib.use('TkAgg')


def main():
    X, y = datasets.make_regression(n_samples=1500, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=300, random_state=42)
    X_train, X_test, y_train, y_test = X[:1200], X[1200:], y[:1200], y[1200:]

    boost_reg = MyBoostReg(n_estimators=20, learning_rate=lambda i: 1.6, loss='MSE', reg=0.01,
                           max_samples=0.3, max_features=0.3, random_state=42, metric='RMSE')
    boost_reg.fit(X_train, y_train, verbose=1, X_eval=X_test, y_eval=y_test, early_stopping=2)

    print(boost_reg.best_score)
    print(len(boost_reg.trees))


if __name__ == '__main__':
    main()
