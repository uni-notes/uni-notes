# Sklearn

## Basics

```python
model = model()

if "n_jobs" in dir(model):
    kwargs["n_jobs"]= -1
if "probability" in dir(model):
    kwargs["probability"]= True

model.set_params(**kwargs)

model.fit(X_train, y_train)
model.predict(X)
```

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

## Time-Series Split

```py
|X||V|O|O|O|
|O|X||V|O|O|
|O|O|X||V|O|
|O|O|O|X||V|
```

X / V are the training / validation sets. "||" indicates a gap (parameter n_gap: int>0) truncated at the beginning of the validation set, in order to prevent leakage effects.

```python
class StratifiedWalkForward(object):
    
    def __init__(self,n_splits,n_gap):
        self.n_splits = n_splits
        self.n_gap = n_gap
        self._cv = StratifiedKFold(n_splits=self.n_splits+1,shuffle=False)
        return
    
    def split(self,X,y,groups=None):
        splits = self._cv.split(X,y)
        _ixs = []
        for ix in splits: 
            _ixs.append(ix[1])
        for i in range(1,len(_ixs)): 
            yield tuple((_ixs[i-1],_ixs[i][_ixs[i]>_ixs[i-1][-1]+self.n_gap]))
            
    def get_n_splits(self,X,y,groups=None):
        return self.n_splits
```

Note that the datasets may not be perfectly stratified afterwards, cause of the truncation with n_gap.

## Regression with Custom Loss Function

[using scipy](../Scipy/04_Regression_with_Custom_Loss_Function.md) 

## Decision Boundary

```python
x_min, x_max = X[:, 0].min(), X[:,0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
resolution = 100

x = np.linspace(x_min - 0.1, x_max + 0.1, resolution)
y = np.linspace(y_min - 0.1, y_max + 0.1, resolution)
```

```python
xx, yy = np.meshgrid(x, y)
```

```python
x_in = np.c_[xx.ravel(), yy.ravel()]
y_pred = model.predict(x_in).reshape(xx.shape)
```

```python
plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7 )
plt.scatter(X[:,0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
```

![img](./assets/1*nsC6mgj-WhjZ7PN0TBcEkg.png)

## SVM

- LinearSVC: Primal
- SVC: Dual

## Multi-Level Model

```python
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

X, y = make_regression(n_samples=100, n_features=10, noise=10.0, random_state=42)

class Regressor(BaseEstimator):
    def __init__(self, n_features):
        self.n_features = n_features
        self.linear_reg = LinearRegression()
        self.boosting_reg = GradientBoostingRegressor(
            n_estimators=1,
            init="zero",
            random_state=42
        )

    def fit(self, X, y):
        self.linear_reg.fit(X=X[:, :self.n_features], y=y)
        y_pred = self.linear_reg.predict(X=X[:, :self.n_features])
        residual = y - y_pred
        self.boosting_reg.fit(X=X[:, self.n_features:], y=residual)
        return self

    def predict(self, X):
        y_pred_linear_reg = self.linear_reg.predict(X=X[:, :self.n_features])
        residual = self.boosting_reg.predict(X=X[:, self.n_features:])
        y_pred = y_pred_linear_reg + residual
        return y_pred


union = FeatureUnion(
    transformer_list=[("ss", StandardScaler()),
                      ("qt", QuantileTransformer(n_quantiles=2))]
)

X_union = union.fit_transform(X=X)

n_features = 10
pipe = Pipeline(
    steps=[
      ("reg", Regressor(n_features=n_features))
    ]
).fit(X=X_union, y=y)
y_pred = pipe.predict(X=X_union)
```

