import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
import math
import time

from sklearn import *

from gradient_boosting_classification import *
from principal_component_analysis import *

matplotlib.use('TkAgg')


def main():
    X, y = datasets.make_classification(n_samples=1500, n_features=10, n_informative=3, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=300, random_state=42)
    X_train, X_test, y_train, y_test = X[:1200], X[1200:], y[:1200], y[1200:]

    boost_clf = MyBoostClf(n_estimators=20, learning_rate=0.5, metric='precision', reg=0.001,
                           max_features=0.3, max_samples=0.3, max_depth=3, bins=8)
    boost_clf.fit(X_train, y_train, X_eval=X_test, y_eval=y_test, early_stopping=2, verbose=1)

    print(boost_clf.best_score)


if __name__ == '__main__':
    main()
