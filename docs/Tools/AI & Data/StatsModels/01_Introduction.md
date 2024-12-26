# Introduction

## Imports

```python
from statsmodels import api as sm
from statsmodels.formula import api as smf
```

## Transformation

```python
X = sm.add_constant(X) # add intercept
```

## Train

### Equation

```python
equation = "y ~ x"
equation = "y ~ x - 1" # remove intercept

# multi-variate
equation = "y ~ x + x^2"
equation = "y ~ x_1 + x_2"

# non-linear
equation = "np.log(y) ~ t - 1"

# embedding external function - i don't like this
equation = "y ~ x_1 + np.log(x_2)"
equation = "y ~ x_1 + custom_function(x_2)"

# categorical
equation = "y ~ x_1 + C(x_categorical)"

# interactions
equation = "y ~ x_1 * x_2" # x_1 + x_2 + x_1:x_2
equation = "y ~ x_1:x_2" # only x_1:x_2
```

```python
model = smf.ols(equation, data = df)
```

### Regular Method

```python
model = sm.OLS(y, X)
```

### Regular Method with Equation

```python
import patsy
y, X = patsy.dmatrices(
	equation,
	data = df,
	return_type = "dataframe"
)
# regular method
```

## Print Results

```python
result = model.fit(exog = X, endog = y)

alpha = 0.05
sig = alpha/X.shape[0] # bonferroni-correction
res.summary(alpha = sig)

res.predict(X_test) # sf
res.predict(df_test) # smf
```


## Confidence Intervals

```python
from statsmodels.sandbox.regression import predstd

std, upper, lower = predstd.wls_prediction_std(model)
```

## Sklearn Wrapper

```python
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class SMWrapper(BaseEstimator):
    """
    A universal sklearn-style wrapper for statsmodels regressors
    """
    
    def __init__(self, estimator, fit_intercept=True, **init_params):
        self.estimator = estimator
        self.fit_intercept = fit_intercept
        self.init_params = init_params
    
    def fit(self, X, y, **fit_params):

        if self.fit_intercept:
            X = sm.add_constant(X)
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.estimator_ = self.estimator(
	        exog = X,
	        endog = y,
	        **self.fit_params
	    )
        self.results_ = self.estimator_.fit()
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'estimator_')

        # Input validation
        X = check_array(X)

        if self.fit_intercept:
            X = sm.add_constant(X)

        return self.results_.predict(X)

    def summary(self, **summary_params):
        return self.results_.summary(**summary_params)

class SMRegressor(RegressorMixin, SMWrapper):
    def __init__(self, estimator, fit_intercept=True, **init_params):
        super().__init__(estimator, fit_intercept, **init_params)

class SMClassifier(ClassifierMixin, SMWrapper):
    def __init__(self, estimator, fit_intercept=True, **init_params):
        super().__init__(estimator, fit_intercept, **init_params)
```

```python
# Model Definition
model = SMRegressor(sm.OLS)
# model = SMRegressor(sm.GLS, sigma = sigma)

# Training
model.fit(X, y)

alpha = 0.05
sig = alpha/X.shape[1] # bonferroni-correction
print(model.summary(alpha=sig))

# Inference
model.predict(X)
```