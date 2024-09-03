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
    # boost_reg = MyBoostReg()
    # boost_reg.fit(X_train, y_train)
    #
    # preds = boost_reg.predict(X_test)

    X, y = datasets.make_classification(n_samples=170, n_features=10, n_informative=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=20, random_state=42)

    bag_clf = MyBaggingClf(estimator=MyKNNClf(), max_samples=1.0, oob_score='accuracy')
    bag_clf.fit(X_train, y_train)

    preds = bag_clf.predict_proba(X_test)

    print(bag_clf.oob_score_)


if __name__ == '__main__':
    main()
