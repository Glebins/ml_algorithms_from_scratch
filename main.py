import functools

import pandas as pd
import numpy as np
import math
import time

from random_forest_classification import *

from sklearn import *


def main():
    # path_to_datasets = "C:/Users/nedob/Programming/Data Science/Datasets/"
    # df = pd.read_csv(path_to_datasets + 'banknote_authentication/data.zip', header=None)
    # df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    # X, y = df.iloc[:, :4], df['target']

    X, y = datasets.make_classification(n_samples=150, n_features=10, n_informative=3, random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=None)
    y_test.index = range(len(y_test))
    y_train.index = range(len(y_train))

    rf_clf = MyForestClf(n_estimators=6, max_depth=5, max_features=0.5, max_samples=0.5, oob_score='roc_auc')
    rf_clf.fit(X, y)

    print(rf_clf.oob_score_)


if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     path_to_datasets = "C:/Users/nedob/Programming/Data Science/Datasets/"
#     df = pd.read_csv(path_to_datasets + 'banknote_authentication/data.zip', header=None)
#     df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
#     X, y = df.iloc[:, :4], df['target']
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=None)
#     y_test.index = range(len(y_test))
#     y_train.index = range(len(y_train))
#
#     rf_clf = MyForestClf(n_estimators=100, max_depth=17, max_leafs=30, max_features=0.7, max_samples=0.3, criterion='gini', is_parallel_fit=True,
#                          random_state=None)
#
#     rf_clf.fit(X_train, y_train)
#
#     pred = rf_clf.predict(X_test, type='mean')
#
#     print((pred == y_test).mean())
#     print(rf_clf.fi)

# data = datasets.load_diabetes(as_frame=True)
# X, y = data['data'], data['target']
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
#
# start_time = time.time()
#
# rf_reg = MyForestReg(n_estimators=100, max_depth=14, max_features=0.5, oob_score='r2')
#
# rf_reg.fit(X_train, y_train)
#
# print(f"fit's during is {time.time() - start_time}")
# start_time = time.time()
#
# prediction = rf_reg.predict(X_test)
#
# print(f"prediction's during is {time.time() - start_time}")
