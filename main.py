import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
import math
import time

from bagging_regression import *
from knn_regression import *

from sklearn import *

matplotlib.use('TkAgg')


def main():
    # data = datasets.load_diabetes(as_frame=True)
    # X, y = data['data'], data['target']
    #
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    rs = None

    X, y = datasets.make_regression(n_samples=170, n_features=14, n_informative=10, noise=15, random_state=rs)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # knn_reg = KNNReg()
    #
    # knn_reg.fit(X_train, y_train)
    # knn_reg.predict(X_test)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=20, random_state=rs)

    start_time = time.time()

    bg = MyBaggingReg(estimator=MyTreeReg(max_depth=20, max_leafs=50), max_samples=0.3, n_estimators=100, random_state=rs,
                      oob_score='r2')
    bg.fit(X_train, y_train)

    print(f"Parallel: {time.time() - start_time}")

    pred = bg.predict(X_test)

    print(pred.sum())
    print(bg.oob_score_)


    start_time = time.time()

    bg = MyBaggingReg(estimator=MyTreeReg(max_depth=20, max_leafs=50), max_samples=0.3, n_estimators=100, random_state=rs,
                      oob_score='r2', is_parallel_fit=False)
    bg.fit(X_train, y_train)

    print(f"Sequential: {time.time() - start_time}")

    pred = bg.predict(X_test)

    print(pred.sum())
    print(bg.oob_score_)


if __name__ == '__main__':
    main()
