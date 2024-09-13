import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from DBSCAN_clustering import *

from sklearn import *

matplotlib.use('TkAgg')

centers = 3
X, y = datasets.make_blobs(n_samples=70, centers=centers, n_features=2, cluster_std=2.5, random_state=None)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

dbscan = MyDBSCAN(eps=1.5, min_samples=3, metric='euclidean')
points, status = dbscan.fit_predict(X, True)

print(pd.Series(points).value_counts().sort_index())
print("1 - outliers, 2 - border, 3 - node")
print(dbscan.clusters_number)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(x=X['col_0'], y=X['col_1'], hue=points, palette='viridis', ax=axes[0])

axes[0].set_title('2D Points Clustered')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

sns.scatterplot(x=X['col_0'], y=X['col_1'], hue=status, palette='viridis', ax=axes[1])

axes[1].set_title('2D Points Clustered')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

plt.show()