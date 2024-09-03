# Introduction

```python
# define model
regressor = RandomForestRegressor()

# transform to tabular form
features_no_of_lags = 3 # {1, 2, 3}
forecaster = make_reduction(
	regressor,
	window_length = features_no_of_lags,
	strategy = "recursive"
)
```

```python
forecast_horizon = np.arange(1, 5)

cv_expanding = ExpandingWindowSplitter(
	initial_window = 24*10,
	step_length = 24,
	fh = forecast_horizon,
)

cv_expanding = SlidingWindowSplitter(
	window_length = 24*10,
	step_length = 24,
	fh = forecast_horizon,
)
```

```python
results = evaluate(
	forecaster = forecaster,
	y = y_train,
	cv = cv,
	return_data = True,
	strategy = "refit" # ["refit", "update"]
)
```
