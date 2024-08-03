import pandas as pd
import numpy as np
import math

from linear_regression import *


def func(xx):
    return np.array(round(0.1 * np.log(np.sin(xx[0])) + 0.7, 3))


X = construct_X([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, math.pi / 2, 2, 2.3, 2.5, 2.7, 3])
y = construct_y([0.47, 0.538, 0.578, 0.606, 0.626, 0.643, 0.656, 0.667, 0.7, 0.69, 0.671, 0.649, 0.615, 0.504])

X[0] = np.log(np.sin(X[0]))

lin_reg = LinearReg(metric='r2')

lin_reg.train_X = X
lin_reg.train_y = y

# todo applying functions as a class method

weights_ideal = lin_reg.get_solution()

print(lin_reg.get_metric())

print(lin_reg.get_pretty_string_of_points(in_a_row=True, include_zero=True))

print(lin_reg.get_pretty_str_of_result(round_to=3, variables=['y', 'x', 'x ^ 2']))

print(lin_reg.weights)
