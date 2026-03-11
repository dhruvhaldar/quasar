import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class UnsupervisedModels:
    @staticmethod
    def train_kmeans(X, n_clusters=3, random_state=42):
        X = np.array(X)
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        model.fit(X)

        return {
            'cluster_centers': model.cluster_centers_.tolist(),
            'labels': model.labels_.tolist()
        }

    @staticmethod
    def train_knn(X, y, n_neighbors=5, weights='uniform'):
        X = np.array(X)
        y = np.array(y)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        model.fit(X, y)

        # create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist()
        }
