# Sklearn

### 

## PCA

```python
from sklearn.decomposition import PCA

df = pd.DataFrame(data=np.random.normal(0, 1, (20, 10)))

pca = PCA(n_components=5)
pca.fit(df)

pca.components_
```
## Stratified Sampling
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, 
  test_size = 0.5,
  random_state = 0,
  stratify = y
)
```
## Save & Load Model
## Pickle
```python
import pickle
file_name = "model.pkl"

## save
with open(file_name, "wb") as f:
    pickle.dump(model, f)

#load
with open(file_name, "rb") as f:
  model = pickle.load(f)
```
## Json
```python
## save
model.save_model("model.json")

## load
model_new = xgb.XGBRegressor()
model_new.load_model("model.json")
```
## Pipelines
## What?
Systematic organization of required operations
## Parts
### Transformer
filter and/or modify data
`fit()` and `transform()`
### Estimator
Learn from data
`fit()` and `predict()`
## Implementation
### Libraries
```python
from sklearn.pipeline import make_pipeline
##  just `Pipeline` involves naming each step

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
### Pipeline Used Here
Data Preprocessing by using Standard Scaler
Reduce Dimension using PCA
Apply Classifier
### Initializing Pipelines
```python
pipeline_lr = make_pipeline(
  StandardScaler(),
  LogisticRegression()
)

## more controlled way
pipeline_dt = Pipeline([
  ('scaler',StandardScaler()),
  ('classifier',DecisionTreeClassifier())
])
```
```python
pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest]

## Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {
  0: 'Logistic Regression',
  1: 'Decision Tree'
}

best_accuracy = 0.0
best_classifier = 0
best_pipeline=""
```
### Pipeline Parameters

```python
pipe.get_params()
```

### Training/Fitting

```python
## Fit the pipelines
for pipe in pipelines:
  pipe.fit(X_train, y_train)
    ## pipe.fit(X_train, y_train, classifier__sample_weight=1)
```
### Results
```python
for i,model in enumerate(pipelines):
    print(
      pipe_dict[i], "Test Accuracy:", model.score(X_test,y_test)
    )
```
```python
for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_classifier=i
        best_pipeline=model

print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))
```
## Change Loss_Cost Function
```python
def custom_loss(y_true, y_pred):
    fn_penalty = 5 ## penalty for false negatives
    fp_penalty = 1 ## penalty for false positives

    ## calculate true positives, false positives, and false negatives
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    ## calculate loss
    loss = fp_penalty * fp + fn_penalty * fn

    return loss

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(loss=custom_loss)
```
## Custom Ensembling
Voting
Stacking
## Hyperparameter Tuning

```python
## create the Pipeline
pipe = Pipeline(
  [('preprocessor', ct), ('classifier', clf1)],
  memory = "cache_name" # cache results, especially useful for grid-search
)

## create the parameter dictionary for clf1
params = [
  dict(
    preprocessor__vectorizer__ngram_range = [(1, 1), (1, 2)],
    classifier__penalty = ['l1', 'l2'],
    classifier = [my_random_forest]
  ),
  dict(
    preprocessor__vectorizer__ngram_range = [(1, 1), (1, 2)],
    classifier__n_estimators = [100, 200],
    classifier = [my_decision_tree]
  )
]
```
```python
grid = GridSearchCV(
  pipe,
  param_grid = params,
  cv = 5,
  refit = False, # True forces to refit best model for the entire dataset at the end; pointless if you only want cv results
  n_jobs = -1,
  memory = "grid_search" # caching
)

grid.fit(X, y)

print(grid.best_params_)
```

## Custom Estimator

using scipy  
```python
class CustomRegressionModel(BaseEstimator):
	"""
	All variables inside the Class should end with underscore
	"""
	def __str__(self):
		return str(self.model)
	def __repr__(self):
		return str(self)
	
	def mse(self, pred, true, sample_weight):
		error = pred - true
		
		loss = error**2

		# median is robust to outliers than mean
		cost = np.mean(
			sample_weight * loss
		)

		return cost

	def loss(self, pred, true):
		return self.error(pred, true, self.sample_weight)
	
	def l1(self, params):
		return np.sum(np.abs(params-self.model.initial_guess))
	def l2(self, params):
		return np.sum((params-self.model.initial_guess) ** 2)
	def l3(self, params, alpha=0.5):
		return alpha * self.l1(params) + (1-alpha)*self.l2(params)

	def reg(self, params, penalty_type="l3", lambda_reg_weight = 1.0):
		"""
		lambda_reg_weight = Coefficient of regularization penalty
		"""

		if penalty_type == "l1":
			penalty = self.l1(params)
		elif penalty_type == "l2":
			penalty = self.l2(params)
		elif penalty_type == "l3":
			penalty = self.l3(params)
		else:
			raise Exception

		return lambda_reg_weight * penalty/self.sample_size

	def cost(self, params, X, y):
		pred = self.model.equation(X, *params)
		return self.loss(pred, true=y) #+ self.reg(params) # regularization requires standardised parameters

	def fit(self, X, y, model, method="Nelder-Mead", error = None, sample_weight=None, alpha=0.05):
		check_X_y(X, y) #Using self.X,self.y = check_X_y(self.X,self.y) removes column names

		self.X = X
		self.y = y

		self.n_features_in_ = self.X.shape[1]

		if sample_weight is None or len(sample_weight) <= 1: # sometimes we can give scalar sample weight same for all
			self.sample_size = self.X.shape[0]
		else:
			self.sample_size = sample_weight[sample_weight > 0].shape[0]

		self.sample_weight = (
			sample_weight
			if sample_weight is not None
			else np.full(self.sample_size, 1) # set Sample_Weight as 1 by default
		)

		self.error = (
			error
			if error is not None
			else self.mse
		)

		self.model = model

		params = getfullargspec(self.model.equation).args
		params = [param for param in params if param not in ['self', "x"]]
		
		self.optimization = o.minimize(
			self.cost,
			x0 = self.model.initial_guess,
			args = (self.X, self.y),
			method = method, # "L-BFGS-B", "Nelder-Mead", "SLSQP",
			constraints = [

			],
			bounds = [
				(-1, None) for param in params # variables must be positive
			]
		)

		self.dof = self.sample_size - self.model.k - 1 # n-k-1

		if self.dof <= 0:
			self.popt = [0 for param in params]
			st.warning("Not enough samples")
			return self
		
		success = self.optimization.success
		if success is False:
			st.warning("Did not converge!")

		self.popt = (
			self.optimization.x
		)

		self.rmse = mse(
			self.output(self.X),
			self.y,
			sample_weight = self.sample_weight,
			squared=False
		)

		cl = 1 - (alpha/2)

		if "hess_inv" in self.optimization:
			self.covx = (
				self.optimization
				.hess_inv
				.todense()
			)

			self.pcov = list(
				np.diag(
					self.rmse *
					np.sqrt(self.covx)
				)
			)

			self.popt_with_uncertainty = [
				f"""{{ \\small (
					{round_f(popt, 6)}
					±
					{round_f(stats.t.ppf(cl, self.dof) * pcov.round(2), 2)}
				)}}""" for popt, pcov in zip(self.popt, self.pcov)
			]
		else:
			self.popt_with_uncertainty = [
				f"""{{ \\small {round_f(popt, 5)} }}""" for popt in self.popt
			]

		self.model.set_fitted_coeff(*self.popt_with_uncertainty)
		
		return self
	
	def output(self, X):
		return (
			self.model
			.equation(X, *self.popt)
		)
	
	def get_se_x_cent(self, X_cent):
		return self.rmse * np.sqrt(
			(1/self.sample_size) + (X_cent.T).dot(self.covx).dot(X_cent)
		)
	def get_pred_se(self, X):
		if False: # self.covx is not None: # this seems to be abnormal. check this
			X_cent = X - self.X.mean()
			se = X_cent.apply(self.get_se_x_cent, axis = 1)
		else:
			se = self.rmse
		return se

	def predict(self, X, alpha=0.05):
		check_is_fitted(self) # Check to verify if .fit() has been called
		check_array(X) #X = check_array(X) # removes column names

		pred = (
			self.output(X)
			.astype(np.float32)
		)

		se = self.get_pred_se(X)

		cl = 1 - (alpha/2)

		ci =  stats.t.ppf(cl, self.dof) * se

		return pd.concat([pred, pred+ci, pred-ci], axis=1)
```
```python
model = CustomRegressionModel()
print(model) ## prints latex

model.fit(
  X_train,
  y_train,
  model = Arrhenius(),
  method = "Nelder-Mead"
)
model.predict(X_test)

print(model) ## prints latex with coefficent values
```
## Linear Regression statistical inference
Parameter standard errors
```python
N = len(X)
p = len(X.columns) + 1  ## plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = X.values

beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
print(beta_hat)

y_hat = model.predict(X)
residuals = y.values - y_hat
residual_sum_of_squares = residuals.T @ residuals
sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
for p_ in range(p):
    standard_error = var_beta_hat[p_, p_] ** 0.5
    print(f"SE(beta_hat[{p_}]): {standard_error}")
```
## Parameter confidence intervals

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def get_conf_int(X, y, model, alpha=0.05):

    """
    ## alpha = 0.05 for 95% confidence interval; 0.01 for 99%-CI

    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """

    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = X.copy()
    X_aux.insert(0, 'const', 1)
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - model.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha/2, dof)
    gap = t_val * np.sqrt(mse * var_params)

    return pd.DataFrame({
        'lower': coefs - gap, 'upper': coefs + gap
    }, index=X_aux.columns)


model = LinearRegression().fit(X_train, Y_train)
get_conf_int(X_train, y_train, model, alpha = 0.05)
```
## Mean response confidence intervals

```python
import numpy as np
import pandas as pd
from scipy import stats

X = np.array([
  [0, 10],
  [5, 5],
  [10, 2]
])

X_centered = X - X.mean()

x_pred = np.array(
  [[5, 10]]
)
x_pred_centered = x_pred-X.mean()

n = X.shape[0]
k = X.shape[1]

idk = (
    #(1/n) +
    np.diag(
      x_pred_centered.T
      .dot(
          np.linalg.inv(
              X_centered.T
              .dot(X_centered)
          )
      )
      .dot(x_pred_centered)
    )
)

se_confidence = (
    X.std()
    *
    np.sqrt(
      idk
  )
)
se_prediction = (
    X.std()
    *
    np.sqrt(
      1 + idk
  )
)

alpha = 0.05
dof = n - k
t_val = stats.t.isf(alpha/2, dof)

gap_confidence = t_val * se_confidence
gap_prediction = t_val * se_prediction

print(gap_confidence)
#print(gap_prediction)
```
## Custom Scorer
```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

def mean_error(y, y_pred):
    return np.mean(y_pred - y)
def std_error(y, y_pred):
    return np.std(y_pred - y)

mean_error_scorer = make_scorer(mean_error, greater_is_better=False)
std_error_scorer = make_scorer(mean_error, greater_is_better=False)

model = LinearRegression()
cross_val_score(model, X, y, scoring=mean_error_scorer)
cross_val_score(model, X, y, scoring=std_error_scorer)
```
## Scaling
```python
## demonstrate data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
## load data
data = ...
## create scaler
scaler = MinMaxScaler()
## fit and transform in one step
normalized = scaler.fit_transform(data)
## inverse transform
inverse = scaler.inverse_transform(normalized)
```