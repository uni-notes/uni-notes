# KNN

## KNN Ridge

Useful for one-hot encoded input

```python
class RidgeKNNRegressor(BaseEstimator, RegressorMixin):
	def __init__(self, one_hot_columns, **init_params):
		self.one_hot_columns = one_hot_columns
		self.knn = KNeighborsRegressor(**init_params)
		self.ridge = Ridge(fit_intercept = False)

	def transform_one_hot(self, X):
		X[self.one_hot_columns_idx] = X[self.one_hot_columns_idx] * self.ridge_.coef_[self.one_hot_columns_idx]
		return X
		
	def fit(self, X, y, **fit_params):
		self.one_hot_columns_idx = X.columns.get_indexer(self.one_hot_columns)
		
		self.ridge_ = clone(self.ridge)
		self.ridge_.fit(X, y)
		
		X = self.transform_one_hot(X)
		
		self.knn_ = clone(self.knn)
		self.knn_.fit(X, y, **fit_params)
		
		return self

	def predict(self, X, y=None):
		X = self.transform_one_hot(X)
		return self.knn_.predict(X)
```

## KNN Median

```python
from sklearn.neighbors.regression import KNeighborsRegressor, check_array, _get_weights

class MedianKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.median(_y[neigh_ind], axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred    

X = np.random.rand(100, 1)
y = 20 * X.ravel() + np.random.rand(100)
clf = MedianKNNRegressor().fit(X, y)
print(clf.predict(X[:5]))
# [  2.38172861  13.3871126    9.6737255    2.77561858  17.07392584]
```

## KNN CV

```python
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

class KNeighborsEstimatorCV(BaseEstimator):
	def __init__(self, estimator, cv_estimator, n_jobs=-1, **init_params):
		self.estimator = estimator
		self.cv_estimator = cv_estimator
		self.init_params = init_params
	
	def fit(self, X, y, **fit_params):
		self.pipeline = Pipeline(
	        steps=[
				("graph", KNeighborsTransformer(n_neighbors=max(kwargs["n_neighbors"]))),
			    ("estimator", self.cv_estimator(self.estimator(metric="precomputed"), **self.init_params, n_jobs=self.n_jobs)
				)
		    ],
	        memory="knncv" if "Grid" in self.cv_estimator_.__class__.__name__ else None
	    )

		self.pipeline.fit(self., y, **fit_params)
	
	def predict(self, X, y=None):
		return self.pipeline.predict(X, y)

class KNeighborsRegressorCV(BaseEstimator, RegressorMixin, KNeighborsEstimatorCV):
	def __init__(self, cv_estimator, n_jobs=-1, **init_params):
		return super().__init__(KNeighborsRegressor, cv_estimator, n_jobs=-1, **init_params)

class KNeighborsClassifierCV(BaseEstimator, RegressorMixin, KNeighborsEstimatorCV):
	def __init__(self, cv_estimator, n_jobs=-1, **init_params):
		return super().__init__(KNeighborsClassifier, cv_estimator, n_jobs=-1, **init_params)
```