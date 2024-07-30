import pandas as pd
import numpy as np

import linear_regression
from linear_regression import *


def pretty_print_points(X, y):
    res = ''
    for i in range(len(X.index.values)):
        res += '('
        for j in range(len(X.columns.values)):
            res += f"{X.iloc[i, j]}, "
        res += str(y.iloc[i])
        res += ')\n'

    res = res[:-2]
    res += ')'
    return res


lin_reg = LinearReg(n_iter=100000, learning_rate=0.0001, metric='r2', l1_coef=1, l2_coef=0.1, reg='l1')

np.random.seed(42)

# X = pd.DataFrame(data=np.array([[1, 2, 3, 4], [3, 4, 1, 4]]).T)
# y = pd.Series([2, 3, 1, 4])
X = pd.DataFrame(data=np.random.randint(100, size=(10, 2)))
y = pd.Series(np.random.randint(100, size=(10,)))

print(X, y, sep='\n\n', end='\n\n')

print(pretty_print_points(X, y))

lin_reg.fit(X, y, verbose=10000)
print('\n', lin_reg.get_weights(), sep='', end='\n\n')

res_str = "z = "
chrs = ['', ' * x', ' * y']

for coef, chr in zip(lin_reg.get_weights(), chrs):
    res_str += f"{round(coef, 3)}{chr} + "

res_str = res_str[:-3]
print(res_str, end='\n\n')

beta_ideal = lin_reg.get_solution(X, y)
print(beta_ideal, end='\n\n')

res_str = "z = "
chrs = ['', ' * x', ' * y']

for coef, chr in zip(beta_ideal, chrs):
    res_str += f"{round(coef, 3)}{chr} + "

res_str = res_str[:-3]
print(res_str)

print(lin_reg.get_metric_for_ideal_solution(X, y))
