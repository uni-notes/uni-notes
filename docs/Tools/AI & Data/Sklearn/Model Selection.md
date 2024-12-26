# Model Selection

Trick: Treat Model Class as hyperparameter to tune

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
```

```python
from sklearn.model_selection import RepeatedKFold

inner_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
```

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(
  [
	  ("preprocessor", preprocessing_pipeline),
	  ('model', None) # Set None as placeholder
  ], 
  memory = "cache_name" # cache results, especially useful for grid-search
)
```

## Hyperparameter Tuning

```python
hyperparameter_tuning_params = [
  dict(
    preprocessor__vectorizer__ngram_range = [(1, 1), (1, 2)],
    model = [LinearRegression()],
    model__penalty = ['l1', 'l2'],
  ),
  dict(
    preprocessor__vectorizer__ngram_range = [(1, 1), (1, 2)],
    model = [RandomForestRegressor()],
    model__n_estimators = [1, 2],
  )
]
```

```python
hyperparameter_tuning_search = RandomizedSearchCV(
  pipe,
  param_distributions = hyperparameter_tuning_params,
  n_iter = 10, # only for random search
  cv = inner_cv, # RepeatedKFold
  refit = False, # True forces to refit best model for the entire dataset at the end; pointless if you only want cv results
  n_jobs = -1,
  # memory = "hyperparameter_tuning" # caching; do not use for RandomSearch
)

hyperparameter_tuning_search.fit(X_train_inner_val, y_train_inner_val)
```

```python
results = pd.DataFrame(hyperparameter_tuning_search.cv_results_)
```

Note: [Cross-validation estimators](https://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation) are faster than using the model inside CrossValidation, mainly
- RidgeCV
- LassoCV
- LogisticRegressionCV

```python
alphas = np.logspace(-2, 2, num=10, base=10)

# as fast as Ridge for a single alpha
model = RidgeCV(
	alphas = alphas,
)

# you could nest RidgeCV inside GridSearchCV, using list of alphas as single list item
model = GridSearchCV(
	RidgeCV(),
	params = dict(
		alphas = [alphas],
	),
)
```

## RepeatedSearchCV

 ~~Should also repeat the gridsearch/randomsearch with different random seed.~~ not required as it is only applicable for randomsearch and will end up giving different estimators, which is not goal of model evaluation

```python
class RepeatedSearchCV():
	def __init__(self, search, randomized_steps=None, n_repeats=3, random_state=None):
		self.search = search
		self.randomized_steps = randomized_steps
		self.random_state = random_state
		self.n_repeats = n_repeats

		search_params = self.search.get_params()
		previous_params = search_params[param for param in search_params.keys() if "param" in param] # param_grid or param_distribution
		if randomized_steps is None:
			estimators = [search_parms["estimator"]]
			updated_params = previous_params
			updated_params["random_state"] = [self.random_state + i for i in range(0, n_repeats, 1)]
		else:
			estimators = [
				step_estimator
				for step_id, step_estimator in search_parms["estimator"].steps
				if step_id in self.randomized_steps
				and "random_state" in dir(step_estimator)
			]
			
			updated_params = []
			for previous_param in previous_params:
				for i in range(0, n_repeats, 1):
					temp = previous_param
					temp["random_state"] = self.random_state + i
					updated_params.append(temp)
		self.updated_params = updated_params
		
	def fit(self, X, y):
		self.search_ = clone(self.search)

		updated_kwargs = {}

		for key, value in self.search_.__dir__.keys():
		  if key in ["param_grid", "param_distribution"]:
			  updated_kwargs[key] = self.updated_params

		  if key == "n_iter":
			  updated_kwargs[key] *= self.n_repeats x len(self.randomized_steps)
		
		self.search_.set_params(**updated_kwargs)
		self.search_.fit(X, y)

		for key, value in self.search_.__dict__.items():
		   if key.endswith('_'):
			   setattr(self, key, value)
	

	def predict(self, X, y):
		pass
```

## NestedCV

```python
import time
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class NestedCV():
  def __init__(self, pipeline, params, inner_cv, outer_cv, n_repeats_fit, scoring, hyperparameter_tuning_niter, min_inner_size=None, random_state=0, refit=True):
    self.pipeline = pipeline
    self.params = params
    self.inner_cv = inner_cv
    self.outer_cv = outer_cv
    self.n_repeats_fit = n_repeats_fit
    self.scoring = scoring
    self.hyperparameter_tuning_niter = hyperparameter_tuning_niter
    self.refit = refit
    self.random_state = random_state
    self.min_inner_size = min_inner_size

  def fit(self, X, y):
    check_X_y(X, y)

    best_hyperparameters = []
    validation_metrics = []
    train_metrics = []
    outer_fold = []
    model_classes = []
    tested_params = []

    train_durations = []
    inference_durations = []

    for i, (outer_train_idx, outer_valid_idx) in enumerate(self.outer_cv.split(X, y)):
      
      for param in self.params:
          
          for i, (inner_train_idx, inner_valid_idx) in enumerate(self.outer_cv.split(X, y)):
            if self.min_inner_size is not None and inner_valid_idx.shape[0] < self.min_inner_size:
              return Exception("Not enough samples")

          # inner
          hyperparameter_tuning_search = RepeatedSearchCV(
				RandomizedSearchCV(
		            self.pipeline,
		            param_distributions = param,
		            n_iter = self.hyperparameter_tuning_niter, # only for random search
		            cv = self.inner_cv, # will split into (inner_train_idx, inner_val_idx)
		            refit = True, # True forces to refit best model for the entire dataset at the end; required for nested CV; pointless if you only want cv results
		            n_jobs = -1,
		            scoring = self.scoring,
		            random_state = self.random_state + i,
		        ),
		        n_repeats = 3,
		        random_state = 0
          )
          hyperparameter_tuning_search.fit(X[outer_train_idx], y[outer_train_idx])

          # outer
          outer_fold.append(i+1)
          best_hyperparameter = hyperparameter_tuning_search.best_estimator_
          best_hyperparameters.append(best_hyperparameter)

          tested_params.append(param)

          model_class = best_hyperparameter.steps[-1][1].__class__.__name__
          model_classes.append(model_class)

          train_duration = hyperparameter_tuning_search.refit_time_ / outer_train_idx.shape[0]
          train_durations.append(train_duration)

          train_metric = hyperparameter_tuning_search.score(X[outer_train_idx], y[outer_train_idx])
          train_metrics.append(train_metric)

          inference_start_time = time.time()
          validation_metric = hyperparameter_tuning_search.score(X[outer_valid_idx], y[outer_valid_idx])
          inference_end_time = time.time()

          validation_metrics.append(validation_metric)

          inference_duration = (inference_end_time - inference_start_time) / outer_valid_idx.shape[0]
          inference_durations.append(inference_duration)

    df = (
        pd.DataFrame()
        .assign(
          outer_fold = outer_fold,
          model_class = model_classes,
          model = best_hyperparameters,
          tested_params = tested_params,
          train_metrics = train_metrics,
          validation_metrics = validation_metrics,
          train_duration_per_row = train_durations,
          inference_duration_per_row = inference_durations
        )
    )

    def my_func(x, statistics=["mean", "std"]):
      temp = x.agg(statistics)
      return temp.iloc[0].round(4).astype(str) + " ± " + temp.iloc[1].round(4).astype(str)
    
    summary = (
        df
        .groupby("model_class")
        [["train_metrics", "validation_metrics", "train_duration_per_row", "inference_duration_per_row"]]
        .agg(my_func)
    )

    self.cv_results_ = summary.to_dict()

    if self.refit:

      best_model_class = (
          df
          .groupby("model_class")
          ["validation_metrics"]
          .mean()
          .idxmax()
      )

      best_row = (
          df[
              df["model_class"] == best_model_class
          ]
          .iloc[0]
      )

      best_model = best_row["model"]
      best_params_search = best_row["tested_params"]

      best_model_hyperparameter_search_ = RandomizedSearchCV(
          best_model,
          param_distributions = best_params_search,
          n_iter = self.hyperparameter_tuning_niter, # only for random search
          cv = self.inner_cv, # will split into (inner_train_idx, inner_val_idx)
          refit = True, # True forces to refit best model for the entire dataset at the end; required for nested CV; pointless if you only want cv results
          n_jobs = -1,
          scoring = self.scoring,
          random_state = self.random_state,
          # memory = "hyperparameter_tuning" # caching; do not use for RandomSearch
      )
      best_model_hyperparameter_search_.fit(X, y)
      
	  for key, value in best_model_hyperparameter_search_.__dict__.items():
		   if key.endswith('_'):
			   setattr(self, key, value)
      # self.best_params_ = best_model_hyperparameter_search_.best_params_
      # self.best_estimator_ = best_model_hyperparameter_search_.best_estimator_
      # self.score = best_model_hyperparameter_search_.score

    return self
```

```python
pipeline = Pipeline(
  [
	('model', None) # Set None as placeholder
  ],
#  memory = "cache_name" # cache results, especially useful for grid-search
)

params = [
  dict(
    model = [RandomForestClassifier()],
    model__n_estimators = np.arange(2, 10),
  ),
  dict(
	model = [HistGradientBoostingClassifier()],
    model__max_iter = np.arange(2, 10),
  )
]
```

```python
nestedcv = NestedCV(
    pipeline = pipeline,
    params = params,
    inner_cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=0),
    outer_cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=0),
    inner_search_cv = ,
    outer_search_cv = ,
    final_search_cv = ,
    n_repeats_fit = 3,
    hyperparameter_tuning_niter = 1,
    scoring = "f1",
    random_state = 0,
    refit = True,
)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

```python
nestedcv.fit(X_train, y_train)
pd.DataFrame(nestedcv.cv_results_)

print(nestedcv.score(X_train, y_train))
print(nestedcv.score(X_test, y_test))
```