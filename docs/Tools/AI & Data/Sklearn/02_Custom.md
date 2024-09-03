# Custom 

## Regression with Custom Loss Function

```python
import utils

from sklearn.base import BaseEstimator
#from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    # mean_squared_error as mse,
    root_mean_squared_error as rmse,
    mean_absolute_error as mae,
    r2_score as r2
)

import numpy as np
import pandas as pd
from scipy import optimize as o, stats

from inspect import getfullargspec, getsource
# import copy
```

```python
class CustomRegression(BaseEstimator):
    """
    All variables inside the Class should end with underscore
    """
    def __init__(self, model, method="Nelder-Mead", cost = None, alpha=0, l1_ratio=0.5, maxiter = 500, maxfev = 1_000):
        self.model = model
        self.method = method
        self.cost = cost if cost is not None else self.mse
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.maxiter = maxiter
        self.maxfev = maxfev

    def __str__(self):
        return str(self.model_)
    def __repr__(self):
        return str(self)
    
    def mse(self, pred, true, sample_weight):
        error = pred - true
        
        cost = error**2

        cost = (
            np.mean( # median is robust to outliers than mean
                sample_weight * cost
            )
        )

        return cost
   
    def l1(self, params):
        return np.sum(np.abs(params-self.model_.param_initial_guess))
    def l2(self, params):
        return np.sum((params-self.model_.param_initial_guess) ** 2)
    def l3(self, params, l1_ratio=0.5):
        return (
            l1_ratio * self.l1(params) +
            (1 - l1_ratio) * self.l2(params)
        )

    def regularization_cost(self, params, alpha, penalty_type="l3"):
        """
        Regularization requires standardised parameters
        """
        penalty = get_attr(self, penalty_type)(params, self.l1_ratio)

        return (alpha * penalty)/self.sample_size_

    def cost_total(self, params, X, y):
        pred = self.model_.equation(X, *params)
        cost = self.cost(y, pred, self.sample_weight)

        if self.alpha == 0:
            pass
        else:
            cost += self.regularization_cost(params, self.alpha)
        return cost

    def check_enough_samples(self):
        return self.enough_samples_

    def fit(self, X, y, sample_weight=None):
        check_X_y(X, y) # Using self.X,self.y = check_X_y(self.X,self.y) removes column names

        self.X, self.y = X, y

        self.n_features_in_ = self.X.shape[1]

        self.sample_size_ = self.X.shape[0]
        
        self.sample_weight = (
            sample_weight
            if sample_weight is not None
            else np.full(self.sample_size_, 1) # set Sample_Weight as 1 by default
        )

        self.sample_size_ = self.sample_weight[self.sample_weight > 0].shape[0]

        self.model_ = copy.deepcopy(self.model)

        self.dof_ = self.sample_size_ - self.model_.k # - 1 # n-k-1
        if self.dof_ <= 0:
            self.success_ = False
            self.enough_samples_ = False
            # raise Exception("Not enough samples")
            return self
        else:
            self.enough_samples_ = True

        params_all = getfullargspec(self.model_.equation).args        
        params = [param for param in params_all if param not in ['self', "x"]]

        self.optimization_ = o.minimize(
            self.cost_total,
            x0 = self.model_.param_initial_guess,
            args = (self.X, self.y),
            method = self.method, # "L-BFGS-B", "Nelder-Mead", "SLSQP",
            constraints = self.model_.constraints,
            bounds = self.model_.param_bounds,
            # [
            #     (-1, None) for param in params # variables must be positive
            # ]
            options = dict(
                maxiter = self.maxiter,
                maxfev = self.maxfev,
                # xatol=1e-4,
                # fatol=1e-4,
                # adaptive=False,
            )
        )
        self.success_ = self.optimization_.success

        self.fitted_coeff_ = (
            self.optimization_.x
        )

        self.fitted_coeff_formatted_ = [
            f"""{{ {utils.round_f(popt, 4)} }}""" for popt in self.fitted_coeff_
        ]
            
        self.model_.set_fitted_coeff(*self.fitted_coeff_)
        self.model_.set_fitted_coeff_formatted(*self.fitted_coeff_formatted_)

        self.rmse_response = root_mean_squared_error(
            y,
            self.output(X),
        )
        self.rmse_link = root_mean_squared_error(
            y,
            self.output(X),
        )
        
        return self
    
    def output(self, X):
        return np.array(
            self.model_
            .equation(X, *self.fitted_coeff_)
        )
    
    def get_pred_se(self, X):
        return self.rmse_response

    def predict(self, X):
        check_is_fitted(self, "fitted_coeff_") # Check to verify if .fit() has been called
        check_array(X) #X = check_array(X) # removes column names

        pred = (
            self.output(X)
            .astype(np.float32)
        )

        return pred

    def predict_quantile(self, X, q):
        return self.model_.quantile(X, self.X, self.y, link_distribution_dof=self.dof_, link_distribution_q=q)


class CustomRegressionGrouped(CustomRegression):
    def __str__(self):
        x = ""
        for key, value in self.get_params().items():
            x += f"{str(key)}: {str([utils.round_f(v, 4) for v in list(value)])} \n\n"
        
        return str(x)
    def __init__(self, model, cost, method="Nelder-Mead", group=None):
        super().__init__(model=model, cost=cost, method=method)
        self.group = group
        self.model = model
        self.method = method
        self.cost = cost

        self.optimization_ = dict()
        self.model_ = dict()
        self.enough_samples_ = dict()
        
        self.dof_ = 0
        self.sample_size_ = 0
    
    def get_params(self, how="dict"):
        params = dict()
        for key, value in self.model_.items():
            popt = "fitted_coeff_"
            if popt in dir(value):
                params[key] = getattr(value, popt)
            else:
                params[key] = None
            
        if how == "df":
            params = pd.DataFrame(params).T
        return params
    
    def model_group(self, X, y, model, method, error, sample_weight):
        return (
            CustomRegression(
                model = self.model,
                cost = self.cost,
                method = self.method,
            )
            .fit(
                X,
                y,
                sample_weight
            )
        )
        
    def check_enough_samples(self, how="all"):
        if how == "all":
            enough_samples = True
            for e in self.enough_samples_.values():
                if not e:
                    enough_samples = False
        elif how == "any":
            enough_samples = self.enough_samples_
        else:
            pass
                
        return enough_samples

    def apply_model_multiple_group(self, X, y, group, model, method, cost, sample_weight):
        for g in X[self.group].unique():
            mask = X.eval(f"{self.group} == {g}")
                
            m = self.model_group(
                X[mask],
                y[mask],
                model,
                method,
                cost,
                sample_weight[mask] if sample_weight is not None else sample_weight
            )

            if m.success_:
                self.model_[g] = m
                self.enough_samples_[g] = m.enough_samples_
                self.optimization_[g] = m.optimization_
                self.sample_size_ += m.sample_size_
        
        success = True
        for o in self.optimization_.values():
            if not o.success:
                success = False
                
        self.success_ = success
            
    def fit(self, X, y, sample_weight=None):
        self.model_ = dict()

        self.apply_model_multiple_group(X, y, self.group, self.model, self.method, self.cost, sample_weight)
    def predict(self, X):
        Xs = []
        preds = pd.DataFrame()
        
        for g in X[self.group].unique():
            if g not in self.model_.keys():
                return
            else:
                Xg = X.query(f"{self.group} == {g}")
                index = Xg.index

                pred = self.model_[g].predict(
                    X = Xg.copy().reset_index(drop=True),
                )

                preds = pd.concat([
                    preds, pd.Series(pred, index=index)
                ])
        return preds.sort_index()
    
    def predict_quantile(self, X, q):
        Xs = []
        preds = pd.DataFrame()
        
        for g in X[self.group].unique():
            if g not in self.model_.keys():
                return Exception(f"Model not trained for {g}")
            else:
                Xg = X.query(f"{self.group} == {g}")
                index = Xg.index

                pred = self.model_[g].predict_quantile(
                    X = Xg.copy().reset_index(drop=True),
                    q = q
                )

                preds = pd.concat([
                    preds, pd.Series(pred, index=index)
                ])
        
        return preds.sort_index()
```

```python
curve_fit = CustomRegression(
	model = model_selected,
	cost = regression.LogCosh().cost,
	method = solver
)
print(curve_fit) ## prints latex

curve_fit.fit(
	X_train,
	y_train,
	sample_weight=df_train["Sample_Weight"],
)

print(curve_fit) ## prints latex with coefficent values

pred = curve_fit.predict(X_test)
ll = curve_fit.predict_quantile(X_test, q=0.025)
ul = curve_fit.predict_quantile(X_test, q=0.975)
```

## Holt-Winters

```python
class HoltWinters(BaseEstimator):
    """Scikit-learn like interface for Holt-Winters method."""

    def __init__(self, season_len=24, alpha=0.5, beta=0.5, gamma=0.5):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.season_len = season_len

    def fit(self, series):
        # note that unlike scikit-learn's fit method, it doesn't learn
        # the optimal model paramters, alpha, beta, gamma instead it takes
        # whatever the value the user specified the produces the predicted time
        # series, this of course can be changed.
        beta = self.beta
        alpha = self.alpha
        gamma = self.gamma
        season_len = self.season_len
        seasonals = self._initial_seasonal(series)

        # initial values
        predictions = []
        smooth = series[0]
        trend = self._initial_trend(series)
        predictions.append(smooth)

        for i in range(1, len(series)):
            value = series[i]
            previous_smooth = smooth
            seasonal = seasonals[i % season_len]
            smooth = alpha * (value - seasonal) + (1 - alpha) * (previous_smooth + trend)
            trend = beta * (smooth - previous_smooth) + (1 - beta) * trend
            seasonals[i % season_len] = gamma * (value - smooth) + (1 - gamma) * seasonal
            predictions.append(smooth + trend + seasonals[i % season_len])

        self.trend_ = trend
        self.smooth_ = smooth
        self.seasonals_ = seasonals
        self.predictions_ = predictions
        return self
    
    def _initial_trend(self, series):
        season_len = self.season_len
        total = 0.0
        for i in range(season_len):
            total += (series[i + season_len] - series[i]) / season_len

        trend = total / season_len
        return trend

    def _initial_seasonal(self, series):
        season_len = self.season_len
        n_seasons = len(series) // season_len

        season_averages = np.zeros(n_seasons)
        for j in range(n_seasons):
            start_index = season_len * j
            end_index = start_index + season_len
            season_average = np.sum(series[start_index:end_index]) / season_len
            season_averages[j] = season_average

        seasonals = np.zeros(season_len)
        seasons = np.arange(n_seasons)
        index = seasons * season_len
        for i in range(season_len):
            seasonal = np.sum(series[index + i] - season_averages) / n_seasons
            seasonals[i] = seasonal

        return seasonals

    def predict(self, n_preds=10):
        """
        Parameters
        ----------
        n_preds: int, default 10
            Predictions horizon. e.g. If the original input time series to the .fit
            method has a length of 50, then specifying n_preds = 10, will generate
            predictions for the next 10 steps. Resulting in a prediction length of 60.
        """
        predictions = self.predictions_
        original_series_len = len(predictions)
        for i in range(original_series_len, original_series_len + n_preds):
            m = i - original_series_len + 1
            prediction = self.smooth_ + m * self.trend_ + self.seasonals_[i % self.season_len]
            predictions.append(prediction)

        return predictions
```

```python
def timeseries_cv_score(params, series, loss_function, season_len=24, n_splits=3):
    """
    Iterating over folds, train model on each fold's training set,
    forecast and calculate error on each fold's test set.
    """
    errors = []    
    alpha, beta, gamma = params
    time_series_split = TimeSeriesSplit(n_splits=n_splits) 

    for train, test in time_series_split.split(series):
        model = HoltWinters(season_len, alpha, beta, gamma)
        model.fit(series[train])

        # evaluate the prediction on the test set only
        predictions = model.predict(n_preds=len(test))
        test_predictions = predictions[-len(test):]
        test_actual = series[test]
        error = loss_function(test_actual, test_predictions)
        errors.append(error)

    return np.mean(errors)
```

```python
x = [0, 0, 0]
test_size = 20
data = series.values[:-test_size]
opt = minimize(
  timeseries_cv_score,
  x0=x, 
  args=(data, mean_squared_log_error),
  method='TNC',
  bounds=((0, 1), (0, 1), (0, 1))
)
```

```python
alpha_final, beta_final, gamma_final = opt.x

model = HoltWinters(season_len, alpha_final, beta_final, gamma_final)
model.fit(data)
predictions = model.predict(n_preds=50)

print('original series length: ', len(series))
print('prediction length: ', len(predictions))
```

## Soft Labels/Label Smoothing

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from scipy.special import softmax
import numpy as np

def _log_odds_ratio_scale(X):
    X = np.clip(X, 1e-8, 1 - 1e-8)   # numerical stability
    X = np.log(X / (1 - X))  # transform to log-odds-ratio space
    return X

class FuzzyTargetClassifier(ClassifierMixin, BaseEstimator):
        
    def __init__(self, regressor):
        '''
        Fits regressor in the log odds ratio space (inverse crossentropy) of target variable.
        during transform, rescales back to probability space with softmax function
        
        Parameters
        ---------
        regressor: Sklearn Regressor
            base regressor to fit log odds ratio space. Any valid sklearn regressor can be used here.
        
        '''
        
        self.regressor = regressor
        return
    
    def fit(self, X, y=None, **kwargs):
        #ensure passed y is onehotencoded-like
        y = check_array(y, accept_sparse=True, dtype = 'numeric', ensure_min_features=1)
        self.regressors_ = [clone(self.regressor) for _ in range(y.shape[1])]
        for i in range(y.shape[1]):
            self._fit_single_regressor(self.regressors_[i], X, y[:,i], **kwargs)
        
        return self
    
    def _fit_single_regressor(self, regressor, X, ysub, **kwargs):
        ysub = _log_odds_ratio_scale(ysub)        
        regressor.fit(X, ysub, **kwargs)
        return regressor    
        
    def decision_function(self,X):
        all_results = []
        for reg in self.regressors_:
            results = reg.predict(X)
            if results.ndim < 2:
                results = results.reshape(-1,1)
            all_results.append(results)
        
        results = np.hstack(all_results)                
        return results
    
    def predict_proba(self, X):
        results = self.decision_function(X)
        results = softmax(results, axis = 1)
        return results
    
    def predict(self, X):
        results = self.decision_function(X)
        results = results.argmax(1)
        return results
```