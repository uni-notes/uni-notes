## Performance Optimization

### `n_jobs=-1`

Multi-threading

```python
## !pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()
```

### Config

```python
with sklearn.config_context(
  assume_finite = True
):
  # reduce validation overhead: will not throw a ValueError if X contains NaN or infinity.
	
  pass # do learning/prediction here with reduced validation
```

```python
with sklearn.config_context(
	working_memory = 128 # MB
):
  
  pass # do chunked work here
```

### Model Compression

Linear models

```python
model = SGDRegressor(penalty='elasticnet', l1_ratio=0.25)
model.fit(X_train, y_train)
model.sparsify()
```

### Warm Start

Useful for re-using previous training as initial values

Useful for hyper-parameter tuning

```python
max_estimators = 100

rf = RandomForestClassifier(
  warm_start=True
)

n_estimators = 1
while n_estimators <= max_estimators:
  rf.n_estimators = n_estimators
	rf.fit(X_train, y_train)
 
  n_estimators *= 2
```

The advantage here is that the estimators would already be fit with the previous parameter setting, and with each subsequent call to fit, the model will be starting from the previous parameters, and we're just analyzing if adding new estimators would benefit the model.

### Mini-Batch/Online Learning

```python
model = LinearRegression()

model.partial_fit(data_1)
model.partial_fit(data_2)

```

