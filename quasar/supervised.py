import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score

class SupervisedModels:
    @staticmethod
    def train_svm(X, y, kernel='linear', C=1.0, cv=False):
        X = np.array(X)
        y = np.array(y)
        model = SVC(kernel=kernel, C=C)
        model.fit(X, y)

        # create mesh grid, optimized step size
        step = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                             np.arange(y_min, y_max, step))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        metrics = {}
        if cv:
            cv_results = cross_validate(model, X, y, cv=5, scoring=('accuracy', 'precision', 'recall'))
            metrics = {
                'accuracy': float(np.mean(cv_results['test_accuracy'])),
                'precision': float(np.mean(cv_results['test_precision'])),
                'recall': float(np.mean(cv_results['test_recall']))
            }
        else:
            y_pred = model.predict(X)
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0))
            }

        return {
            'support_vectors': model.support_vectors_.tolist() if hasattr(model, 'support_vectors_') else [],
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist(),
            'metrics': metrics
        }

    @staticmethod
    def train_decision_tree(X, y, criterion='gini', max_depth=None):
        X = np.array(X)
        y = np.array(y)
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        model.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist()
        }

    @staticmethod
    def train_random_forest(X, y, n_estimators=100, max_depth=None):
        X = np.array(X)
        y = np.array(y)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist()
        }

    @staticmethod
    def train_adaboost(X, y, n_estimators=50):
        X = np.array(X)
        y = np.array(y)
        model = AdaBoostClassifier(n_estimators=n_estimators, algorithm='SAMME')
        model.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'Z': Z.tolist()
        }
