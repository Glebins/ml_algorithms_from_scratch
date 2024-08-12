import pandas as pd
import numpy as np
import math

from decision_trees_classification import *

np.random.seed(42)

X = pd.DataFrame(np.round(np.random.random((5, 4)), 1))
y = pd.Series(np.random.randint(0, 2, size=50))

print(X, end="\n\n")

tree_clf = MyTreeClf()

print(get_best_split(X, y))
