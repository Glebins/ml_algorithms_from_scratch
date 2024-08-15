import pandas as pd
import numpy as np
import math

from decision_trees_classification import *

from sklearn import *

path_to_datasets = "C:/Users/nedob/Programming/Data Science/Datasets/"
df = pd.read_csv(path_to_datasets + 'banknote_authentication/data.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:, :4], df['target']

# X, y = datasets.make_classification(n_samples=150, n_features=5, n_informative=3, random_state=42)
# X = pd.DataFrame(X).round(2)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
# y_test.index = range(len(y_test))

tree_clf = MyTreeClf(max_depth=10, min_samples_split=40, max_leafs=21, bins=10, criterion='gini')
tree_clf.fit(X, y)

tree_clf.print_tree()

print(tree_clf.get_debug_info())
print(tree_clf.fi)

# y_predicted = pd.Series(tree_clf.predict(X_test))

# print(metrics.accuracy_score(y_test, y_predicted))
# print(metrics.recall_score(y_test, y_predicted))
# print(metrics.precision_score(y_test, y_predicted))
# print(metrics.f1_score(y_test, y_predicted))








# tree_clf = tree.DecisionTreeClassifier(criterion='log_loss', splitter='best', max_depth=5, min_samples_split=2, max_leaf_nodes=20)
#
# tree_clf.fit(X_train, y_train)
#
# y_predicted = pd.Series(tree_clf.predict(X_test))
#
# print(metrics.accuracy_score(y_test, y_predicted))
# print(metrics.recall_score(y_test, y_predicted))
# print(metrics.precision_score(y_test, y_predicted))
# print(metrics.f1_score(y_test, y_predicted))