from turtledemo.penrose import start

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
    # X, y = datasets.make_regression(n_samples=100, n_features=14, n_informative=10, noise=15, random_state=42)
    # X = pd.DataFrame(X)
    # y = pd.Series(y)
    # X.columns = [f'col_{col}' for col in X.columns]
    #
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=20, random_state=42)
    #
    # boost_reg = MyBoostReg(n_estimators=10, learning_rate=0.1)
    # boost_reg.fit(X_train, y_train)
    #
    # preds = boost_reg.predict(X_test)
    #
    # print(preds)

    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()

    estimator = MyLogReg(n_iter=100000)

    bag_clf = MyBaggingClf(estimator=estimator, n_estimators=10, max_samples=0.2, oob_score='roc_auc',
                           is_parallel_fit=False)
    bag_clf.fit(X_train, y_train)

    print(f'Sequential: {round(time.time() - start_time, 2)} s', bag_clf.oob_score_)
    start_time = time.time()

    bag_clf_1 = MyBaggingClf(estimator=estimator, n_estimators=10, max_samples=0.2, oob_score='roc_auc',
                           is_parallel_fit=True)
    bag_clf_1.fit(X_train, y_train)

    print(f'Parallel: {round(time.time() - start_time, 2)} s', bag_clf_1.oob_score_)


if __name__ == '__main__':
    main()
