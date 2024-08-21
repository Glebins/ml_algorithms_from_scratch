import pandas as pd
import numpy as np
import math
import time

from random_forest_regression import *

from sklearn import *

import multiprocessing


def sum_of_squares(n):
    return sum(i * i for i in range(n))


# Single-core implementation
def single_core_sum_of_squares(n, num_tasks):
    results = []
    for _ in range(num_tasks):
        results.append(sum_of_squares(n))
    return results


# Multi-core implementation
def multi_core_sum_of_squares(n, num_tasks):
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = pool.map(sum_of_squares, [n] * num_tasks)
    return results

# Number of tasks and the size of each task
num_tasks = 5
n = 5_000_000

# Timing the single-core implementation
start_time = time.time()
single_core_results = single_core_sum_of_squares(n, num_tasks)
single_core_time = time.time() - start_time
# Timing the multi-core implementation
start_time = time.time()
multi_core_results = multi_core_sum_of_squares(n, num_tasks)
multi_core_time = time.time() - start_time

# Print the results
print(f"Single-core time: {single_core_time:.2f} seconds")
print(f"Multi-core time: {multi_core_time:.2f} seconds")





# path_to_datasets = "C:/Users/nedob/Programming/Data Science/Datasets/"
# df = pd.read_csv(path_to_datasets + 'banknote_authentication/data.zip', header=None)
# df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
# X, y = df.iloc[:, :4], df['target']

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
