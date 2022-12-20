import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from time import time


class RandomForestMSE:
    def __init__(
        self, n_estimators=9, max_depth=None, feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.n_samples = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        timer = time()
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.n_samples = X.shape[0]
        for i in range(self.n_estimators):
            samples = np.random.choice(self.n_samples, self.n_samples, replace = True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            tree.fit(X[samples, :], y[samples])
            self.trees.append(tree)

        if X_val is not None:
            pred = self.predict(X_val)
            rmse = self.score_rmse(y_val, pred)
            work_time = time() - timer
            return (rmse, work_time)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        n: int
            Count of estimators
        """
        predictions = []
        for i in range(self.n_estimators):
            predictions.append(self.trees[i].predict(X))
        predictions = np.array(predictions)
        res = []
        for i in range(predictions.shape[1]):
            res.append(np.mean(predictions[:, i]))

        return res

    def score_rmse(self, y_true, y_pred):
        return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.shape[0])


class GradientBoostingMSE:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=5, feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.residuals = []
        self.first_leaf = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        timer = time()
        self.first_leaf = y.mean()
        predictions = np.ones(len(y)) * y.mean()

        for i in range(self.n_estimators):
            residuals = y - predictions
            self.residuals.append(residuals)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            tree.fit(X, residuals)
            self.trees.append(tree)
            alp = minimize_scalar(lambda x: np.sum((y - predictions - x * tree.predict(X))**2)).x
            predictions += self.lr * alp * tree.predict(X)

        if X_val is not None:
            pred = self.predict(X_val)
            rmse = self.score_rmse(y_val, pred)
            work_time = time() - timer
            return (rmse, work_time)

    def predict(self, X, n=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        n : int
            Count of estimators
        """
        if n == None:
            n = self.n_estimators
        predictions = np.ones(X.shape[0]) * self.first_leaf
        for i in range(n):
            predictions += self.lr * self.trees[i].predict(X)
        return predictions

    def score_r2(self, y, predicted):
        return 1 - np.sum((predicted - y)**2) / np.sum((y.mean() - y)**2)
    
    def score_rmse(self, y_true, y_pred):
        return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.shape[0])


class RandomForestMSE_n_estimators:
    def __init__(
        self, n_estimators=9, max_depth=None, feature_subsample_size=None, flag_val = 0, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.scores_val = []
        self.scores_train = []
        self.times = []
        self.n_samples = None
        self.flag_val = flag_val

    def fit(self, X, y, X_val, y_val):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        timer = time()
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        self.n_samples = X.shape[0]
        for i in range(self.n_estimators):
            samples = np.random.choice(self.n_samples, self.n_samples, replace = True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            tree.fit(X[samples, :], y[samples])
            self.trees.append(tree)
            self.scores_val.append(self.score_rmse(self.predict(X_val, i + 1), y_val))
            self.scores_train.append(self.score_rmse(self.predict(X, i + 1), y))
            self.times.append(time() - timer)

        if self.flag_val:
            return (np.array(self.scores_train), np.array(self.scores_val))
        return (self.scores_val, self.times)

    def predict(self, X, n=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        n: int
            Count of estimators
        """
        if n == None:
            n = self.n_estimators
        predictions = []
        for i in range(n):
            predictions.append(self.trees[i].predict(X))
        predictions = np.array(predictions)
        res = []
        for i in range(predictions.shape[1]):
            res.append(np.mean(predictions[:, i]))

        return np.array(res)

    def score_rmse(self, y_true, y_pred):
        return np.sqrt(np.sum((y_true - y_pred)**2) / len(y_true))


class GradientBoostingMSE_n_estimators:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=5, feature_subsample_size=None, flag_val = 0, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.flag_val = flag_val
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.scores_train = []
        self.scores_val = []
        self.times = []
        self.residuals = []
        self.first_leaf = None

    def fit(self, X, y, X_val, y_val):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        timer = time()
        self.first_leaf = y.mean()
        predictions = np.ones(len(y)) * y.mean()

        for i in range(self.n_estimators):
            residuals = 2 * (y - predictions)
            self.residuals.append(residuals)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, **self.trees_parameters)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions += self.lr * tree.predict(X)
            self.scores_val.append(self.score_rmse(self.predict(X_val, i + 1), y_val))
            self.scores_train.append(self.score_rmse(self.predict(X, i + 1), y))
            self.times.append(time() - timer)

        if self.flag_val:
            return (np.array(self.scores_train), np.array(self.scores_val))

        return (self.scores_val, self.times)

    def predict(self, X, n=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        n : int
            Count of estimators
        """
        if n == None:
            n = self.n_estimators
        predictions = np.ones(X.shape[0]) * self.first_leaf
        for i in range(n):
            predictions += self.lr * self.trees[i].predict(X)
        return predictions

    def score_r2(self, y, predicted):
        return 1 - np.sum((predicted - y)**2) / np.sum((y.mean() - y)**2)
    
    def score_rmse(self, y_true, y_pred):
        return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.shape[0])
