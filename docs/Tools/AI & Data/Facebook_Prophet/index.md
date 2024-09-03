

# Facebook Prophet

## Limitations

- Basically glorified curve-fitting to time variable
- Tends to overfit
- Does not extrapolate well

## Improve `.fit()`

```python
import numpy as np
import pandas as pd
import time
import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from batch_elastic_net import BatchElasticNetRegression


def make_sine_wave(length: int, n_cycles: int):
    """
    Makes a sine wave given some length and the number of cycles it should go through in that period
    """
    samples = np.linspace(0, length, length)
    return np.sin(2 * np.pi * n_cycles * samples)


def generate_dataset(n_items):
    """
    Generates a time series dataset with weekly frequency for two years. Randomly assigns the yearly, monthly and
    trend values for each item
    """
    year_in_weeks = 104
    yealy_s = make_sine_wave(year_in_weeks, 2)
    monthly_s = make_sine_wave(year_in_weeks, year_in_weeks / 24)
    trend = np.arange(year_in_weeks) / year_in_weeks
    all_ys = []
    for i in range(n_items):
        d = (np.stack([yealy_s, monthly_s, trend], axis=1) * np.random.rand(3)).sum(axis=1) + np.random.rand(year_in_weeks)
        all_ys.append(d + (np.random.rand(len(d))-0.45).cumsum())
    return pd.DataFrame(zip(*all_ys), index = pd.date_range(datetime.datetime(2020, 1, 1), freq='w', periods=len(d)))


def get_changepoint_idx(length, n_changepoints, changepoint_range=0.8):
    """
    Finds the indices of slope change-points using Prophet's logic: assign them uniformly of the first changepoint_range
    percentage of the data
    """
    hist_size = int(np.floor(length * changepoint_range))
    return np.linspace(0, hist_size - 1, n_changepoints+1).round().astype(int)[1:]


def make_changepoint_features(n, changes_idx):
    """
    Creates initial slope and slope change-points features given a length of data and locations of indices.
    The features are 0s for the first elements until their idx is reached, and then they move linearly upwards.
    These features can be used to model a time series with an initial slope and the deltas of change-points.
    """
    linear = np.arange(n).reshape(-1,1)
    feats = [linear]
    for i in changes_idx:
        slope_feat = np.zeros(n)
        slope_feat[i:] = np.arange(0, n-i)
        slope_feat = slope_feat.reshape(-1,1)
        feats.append(slope_feat)
    feat = np.concatenate(feats, axis=1)
    return feat


def run_prophet():
    t = time.time()
    all_prophets_datasets_forecasts = {}
    for name, dataset in data_sets.items():
        all_p_forecast = []
        for i in range(dataset.shape[1]):
            ds = dataset.iloc[:, i].reset_index()
            ds.columns = ['ds', 'y']
            # if uncertainty samples is not None it will take way more time
            m = Prophet(n_changepoints=n_changepoints, changepoint_prior_scale=change_prior, growth='linear',
                        uncertainty_samples=None,
                        yearly_seasonality=True, weekly_seasonality=False, seasonality_prior_scale=seasonality_prior)
            m.fit(ds)
            forecast = m.predict(ds)
            all_p_forecast.append(forecast.yhat)
        all_prophets_datasets_forecasts[name] = pd.DataFrame(zip(*all_p_forecast), index=ds.ds)

    return all_prophets_datasets_forecasts, time.time() - t


def run_batch_linear():
    big_num = 20.  # used as std of prior when it should be uninformative
    p = Prophet()
    t = time.time()
    all_BatchLinear_datasets_forecasts = {}
    for name, dataset in data_sets.items():
        dates = pd.Series(dataset.index)
        dataset_length = len(dataset)
        idx = get_changepoint_idx(dataset_length, n_changepoints)

        seasonal_feat = p.make_seasonality_features(dates, 365.25, 10, 'yearly_sine')
        changepoint_feat = make_changepoint_features(dataset_length, idx) / dataset_length
        feat = np.concatenate([changepoint_feat, seasonal_feat], axis=1)

        n_changepoint_feat = changepoint_feat.shape[1] - 1
        # laplace prior only on changepoints (seasonals get big_num, to avoid l1 regularization on it)
        l1_priors = np.array([big_num] + [change_prior] * n_changepoint_feat + [big_num] * seasonal_feat.shape[1])
        # normal prior on initial slope and on seasonals, and a big_num on changepoints to avoid l2 regularization
        l2_priors = np.array([5] + [big_num] * n_changepoint_feat + [seasonality_prior] * seasonal_feat.shape[
            1])  # normal prior only on seasonal

        # this is how Prophet scales the data before fitting - divide by max of each item
        scale = dataset.max()
        scaled_y = dataset / scale

        blr = BatchElasticNetRegression()
        blr.fit(feat, scaled_y, l1_reg_params=l1_priors, l2_reg_params=l2_priors, as_bayesian_prior=True, verbose=True,
                iterations=1500)

        # get the predictions for the train
        all_BatchLinear_datasets_forecasts[name] = pd.DataFrame(blr.predict(feat) * scale.values, index=dates)

    return all_BatchLinear_datasets_forecasts, time.time() - t


if __name__ == '__main__':
    data_files_names = ['d1', 'd2', 'M5_sample']
    data_sets = {name: pd.read_csv(f'data_files/{name}.csv', index_col=0, parse_dates=True) for name in data_files_names}
    data_sets['randomly_generated'] = generate_dataset(500)

    # can play with these params for both predictors
    change_prior = 0.5
    # the seasonality_prior is an uninformative prior (hardly any regularization), which is the default for Prophet and usually does not require changing
    seasonality_prior = 10
    n_changepoints = 15

    all_prophets_datasets_forecasts, prophet_time = run_prophet()
    all_BatchLinear_datasets_forecasts, batch_time = run_batch_linear()

    print(f'total number of items: {sum([x.shape[1] for x in data_sets.values()])}')
    print(f'Prophet time: {round(prophet_time, 2)}; batch time: {round(batch_time, 2)}')

    # plot examples from datasets (copy to notebook and repeat for different items and datasets)
    name = 'd1'
    batch_preds = all_BatchLinear_datasets_forecasts[name]
    prophet_preds = all_prophets_datasets_forecasts[name]
    orig_data = data_sets[name]

    i = np.random.randint(0, orig_data.shape[1])
    orig_data.iloc[:, i].plot(label='target')
    batch_pred = batch_preds.iloc[:, i]
    prophet_pred = prophet_preds.iloc[:, i]
    prophet_pred.plot(label='prophet')
    batch_pred.plot(label='my_batch')
    plt.title(f'Pearson {round(pearsonr(batch_pred, prophet_pred)[0], 3)}')
    plt.legend()
    plt.show()

    # mean pearson
    all_corrs = {}
    for name in data_sets.keys():
        batch_preds = all_BatchLinear_datasets_forecasts[name]
        prophet_preds = all_prophets_datasets_forecasts[name]
        corrs = []
        for i in range(prophet_preds.shape[1]):
            corrs.append(pearsonr(batch_preds.iloc[:, i], prophet_preds.iloc[:, i])[0])
        all_corrs[name] = np.mean(corrs)
    print(all_corrs)
```

IDK

```python

import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Union

BIG_STD = 20.  # used as std of prior when it should be uninformative (when we do not wish to regularize at all)


def to_tensor(x):
    return torch.from_numpy(np.array(x)).float()


class BatchElasticNetRegression(object):
    """
    Elastic net for the case where we have multiple targets (y), all to be fitted with the same features (X).
    Learning all items in parallel, in a single "network" is more efficient then iteratively fitting a regression for
    each target.
    Allows to set different l1 and l2 regularization params for each of the features.
    Can also be used to estimate the MAP of a Bayesian regression with Laplace or Normal priors instead of L1 and L2.
    """
    def __init__(self):
        self.coefs = None
        self.intercepts = None

    def fit(self,
            X, y,
            l1_reg_params: Optional[Union[np.array, float]] = None,
            l2_reg_params: Optional[Union[np.array, float]] = None,
            as_bayesian_prior=False, iterations=500, verbose=True, lr_rate=0.1):
        """
        Fits multiple regressions. Both X and y are 2d matrices, where X is the common features for all the targets,
        and y contains all the concatenated targets.
        If as_bayesian_prior==False then the l1 and l2 reg params are regularization params
        If as_bayesian_prior==True then l1 is treated as the std of the laplace prior and l2 as the std for the normal
        prior.
        The reg params / std of priors can either be a single value for all features, or set a different regularization
        or prior for each feature separately. e.g. if we have 3 features, l1_reg_params can be [0.5, 1.2, 0] to set
        regularization for each.

        TODO:
        Add normalization before fitting
        Requires more work on the optimizer to be faster
        """
        n_items = y.shape[1]
        k_features = X.shape[1]
        n_samples = X.shape[0]

        # TODO: if l1_reg_params is None just don't calculate this part of the loss, instead of multiplying by 0
        if l1_reg_params is None:
            l1_reg_params = BIG_STD if as_bayesian_prior else 0.
        if type(l1_reg_params) == float:
            l1_reg_params = [l1_reg_params] * k_features
        if l2_reg_params is None:
            l2_reg_params = BIG_STD if as_bayesian_prior else 0.
        if type(l2_reg_params) == float:
            l2_reg_params = [l2_reg_params] * k_features

        assert len(l1_reg_params) == len(l2_reg_params) == k_features, 'Regularization values must match X.shape[1]'
        if as_bayesian_prior:
            assert 0 not in l1_reg_params and 0 not in l2_reg_params, 'Cannot have 0 prior'

        # convert to tensors and set initial params
        t_features = to_tensor(X)
        t_target = to_tensor(y)
        learned_coefs = torch.rand(k_features, n_items, requires_grad=True)
        learned_intercepts = torch.rand(n_items, requires_grad=True)
        # TODO: or auto-estimate initial sigma based on data std?
        est_sigma = torch.ones(n_items)
        if as_bayesian_prior:
            # If the params are priors then they must become a matrix, not a simple vector - because the conversion
            # depends on the sigma of errors for each target y. The actual regularization params will be different
            # for each item based on its sigma.
            t_l1_reg_params = to_tensor(np.stack([l1_reg_params] * n_items, axis=1))
            l1_alpha = self.calc_l1_alpha_from_prior(est_sigma, t_l1_reg_params, n_samples)
            t_l2_reg_params = to_tensor(np.stack([l2_reg_params] * n_items, axis=1))
            l2_alpha = self.calc_l2_alpha_from_prior(est_sigma, t_l2_reg_params, n_samples)
        else:
            l1_alpha = to_tensor(l1_reg_params)
            l2_alpha = to_tensor(l2_reg_params)

        # TODO: add scheduler for learning rate
        optimizer = optim.Adam([learned_coefs, learned_intercepts], lr_rate)

        for i in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            res = torch.matmul(t_features, learned_coefs) + learned_intercepts
            diff_loss = (1 / (2 * n_samples)) * ((res - t_target) ** 2).sum(axis=0)

            if as_bayesian_prior:
                reg_loss = (l1_alpha * learned_coefs.abs()).sum(axis=0) + (l2_alpha * learned_coefs ** 2).sum(axis=0)
            else:
                reg_loss = torch.matmul(l1_alpha, learned_coefs.abs()) + torch.matmul(l2_alpha, learned_coefs ** 2)

            loss = (diff_loss + reg_loss).sum()

            loss.backward()
            optimizer.step()
            if as_bayesian_prior and i % 50 == 0:
                # if the params are the priors - we must convert them to the equivalent l1/l2 loss params.
                # This conversion depends on the final sigma of errors of the forecast, which is unknown until we
                # have a forecast using those same params... We iteratively improve our estimate of sigma and
                # re-compute the corresponding regularization params based on those sigmas.
                # The sigma is per target in y, therefore the l1/l2 params are per item.
                est_sigma = (res - t_target).std(axis=0).detach()
                l1_alpha = self.calc_l1_alpha_from_prior(est_sigma, t_l1_reg_params, n_samples)
                l2_alpha = self.calc_l2_alpha_from_prior(est_sigma, t_l2_reg_params, n_samples)

            if i % 50 == 0 and verbose:
                print(loss)
            # TODO: early stopping if converges

        self.coefs = learned_coefs.detach().numpy()
        self.intercepts = learned_intercepts.detach().numpy()

    def predict(self, X):
        return X @ self.coefs + self.intercepts

    @staticmethod
    def calc_l1_alpha_from_prior(est_sigma, b_prior, n_samples):
        """
        Converts from the std of a Laplace prior to the equivalent L1 regularization param.
        The conversion formula is divided by 2*n_samples since we divided the diff_loss by 2*n_samples as well,
        to match sklearn's implementation of Lasso.
        """
        return est_sigma ** 2 / (b_prior * n_samples)

    @staticmethod
    def calc_l2_alpha_from_prior(est_sigma, b_prior, n_samples):
        return est_sigma ** 2 / (b_prior ** 2 * 2 * n_samples)
```

## Improve `.predict()`

01/25/23 edit: FB engineers integrated the solution in this post into the package in version release 1.1.2. Due to some implementation differences it is still slightly slower than the solution in the end of this post, but not significantly.

```python
import numpy as np
import pandas as pd

from fbprophet import Prophet


def _make_historical_mat_time(deltas, changepoints_t, t_time, n_row=1):
    """
    Creates a matrix of slope-deltas where these changes occured in training data according to the trained prophet obj
    """
    diff = np.diff(t_time).mean()
    prev_time = np.arange(0, 1 + diff, diff)
    idxs = []
    for changepoint in changepoints_t:
        idxs.append(np.where(prev_time > changepoint)[0][0])
    prev_deltas = np.zeros(len(prev_time))
    prev_deltas[idxs] = deltas
    prev_deltas = np.repeat(prev_deltas.reshape(1, -1), n_row, axis=0)
    return prev_deltas, prev_time


def prophet_logistic_uncertainty(
    mat: np.ndarray,
    deltas: np.ndarray,
    prophet_obj: Prophet,
    cap_scaled: np.ndarray,
    t_time: np.ndarray,
):
    """
    Vectorizes prophet's logistic growth uncertainty by creating a matrix of future possible trends.
    """

    def ffill(arr):
        mask = arr == 0
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return arr[np.arange(idx.shape[0])[:, None], idx]

    k = prophet_obj.params["k"][0]
    m = prophet_obj.params["m"][0]
    n_length = len(t_time)
    #  for logistic growth we need to evaluate the trend all the way from the start of the train item
    historical_mat, historical_time = _make_historical_mat_time(deltas, prophet_obj.changepoints_t, t_time, len(mat))
    mat = np.concatenate([historical_mat, mat], axis=1)
    full_t_time = np.concatenate([historical_time, t_time])

    #  apply logistic growth logic on the slope changes
    k_cum = np.concatenate((np.ones((mat.shape[0], 1)) * k, np.where(mat, np.cumsum(mat, axis=1) + k, 0)), axis=1)
    k_cum_b = ffill(k_cum)
    gammas = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        x = full_t_time[i] - m - np.sum(gammas[:, :i], axis=1)
        ks = 1 - k_cum_b[:, i] / k_cum_b[:, i + 1]
        gammas[:, i] = x * ks
    # the data before the -n_length is the historical values, which are not needed, so cut the last n_length
    k_t = (mat.cumsum(axis=1) + k)[:, -n_length:]
    m_t = (gammas.cumsum(axis=1) + m)[:, -n_length:]
    sample_trends = cap_scaled / (1 + np.exp(-k_t * (t_time - m_t)))
    # remove the mean because we only need width of the uncertainty centered around 0
    # we will add the width to the main forecast - yhat (which is the mean) - later
    sample_trends = sample_trends - sample_trends.mean(axis=0)
    return sample_trends


def _make_trend_shift_matrix(mean_delta: float, likelihood: float, future_length: float, k: int = 10000) -> np.ndarray:
    """
    Creates a matrix of random trend shifts based on historical likelihood and size of shifts.
    Can be used for either linear or logistic trend shifts.
    Each row represents a different sample of a possible future, and each column is a time step into the future.
    """
    # create a bool matrix of where these trend shifts should go
    bool_slope_change = np.random.uniform(size=(k, future_length)) < likelihood
    shift_values = np.random.laplace(0, mean_delta, size=bool_slope_change.shape)
    mat = shift_values * bool_slope_change
    n_mat = np.hstack([np.zeros((len(mat), 1)), mat])[:, :-1]
    mat = (n_mat + mat) / 2
    return mat


def add_prophet_uncertainty(
    prophet_obj: Prophet,
    forecast_df: pd.DataFrame,
    using_train_df: bool = False,
):
    """
    Adds yhat_upper and yhat_lower to the forecast_df used by fbprophet, based on the params of a trained prophet_obj
    and the interval_width.
    Use using_train_df=True if the forecast_df is not for a future time but for the training data.
    """
    assert prophet_obj.history is not None, "Model has not been fit"
    assert "yhat" in forecast_df.columns, "Must have the mean yhat forecast to build uncertainty on"
    interval_width = prophet_obj.interval_width

    if using_train_df:  # there is no trend-based uncertainty if we're only looking on the past where trend is known
        sample_trends = np.zeros(10000, len(forecast_df))
    else:  # create samples of possible future trends
        future_time_series = ((forecast_df["ds"] - prophet_obj.start) / prophet_obj.t_scale).values
        single_diff = np.diff(future_time_series).mean()
        change_likelihood = len(prophet_obj.changepoints_t) * single_diff
        deltas = prophet_obj.params["delta"][0]
        n_length = len(forecast_df)
        mean_delta = np.mean(np.abs(deltas)) + 1e-8
        if prophet_obj.growth == "linear":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=10000)
            sample_trends = mat.cumsum(axis=1).cumsum(axis=1)  # from slope changes to actual values
            sample_trends = sample_trends * single_diff  # scaled by the actual meaning of the slope
        elif prophet_obj.growth == "logistic":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=1000)
            cap_scaled = (forecast_df["cap"] / prophet_obj.y_scale).values
            sample_trends = prophet_logistic_uncertainty(mat, deltas, prophet_obj, cap_scaled, future_time_series)
        else:
            raise NotImplementedError

    # add gaussian noise based on historical levels
    sigma = prophet_obj.params["sigma_obs"][0]
    historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
    full_samples = sample_trends + historical_variance
    # get quantiles and scale back (prophet scales the data before fitting, so sigma and deltas are scaled)
    width_split = (1 - interval_width) / 2
    quantiles = np.array([width_split, 1 - width_split]) * 100  # get quantiles from width
    quantiles = np.percentile(full_samples, quantiles, axis=0)
    # Prophet scales all the data before fitting and predicting, y_scale re-scales it to original values
    quantiles = quantiles * prophet_obj.y_scale

    forecast_df["yhat_lower"] = forecast_df.yhat + quantiles[0]
    forecast_df["yhat_upper"] = forecast_df.yhat + quantiles[1]
```

```python
p = Prophet(uncertainty_samples=None) # tell Prophet not to create the interval by itself
p = p.fit(training_df)

# set to your number of periods and freq
forecast_df = p.make_future_dataframe(periods=10, freq='W', include_history=False)
training_df = p.predict(training_df)
forecast_df = p.predict(forecast_df)
add_prophet_uncertainty(p, training_df, using_train_df=True)
add_prophet_uncertainty(p, forecast_df)
```

