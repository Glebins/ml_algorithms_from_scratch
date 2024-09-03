import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
import math
import time

from bagging_classification import *

from sklearn import *

matplotlib.use('TkAgg')


def main():
    X, y = datasets.make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    bag_clf = MyBaggingClf(estimator=MyLogReg())
    bag_clf.fit(X, y)

    print(bag_clf.estimators)


if __name__ == '__main__':
    main()
