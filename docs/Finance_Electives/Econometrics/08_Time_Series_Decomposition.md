# Time Series Decomposition

|                                                     |                                                              | Advantages                                                   | Disadvantages                                                |
| --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Classical                                           |                                                              | Easy to understand & interpret                               | 1. Estimate of trend is unavailable in the first few and last few observations<br/>2. Assumes that seasonal component repeats<br/>3. Not robust to outliers due to usage of means |
| X-11                                                |                                                              | 1. Relatively robust to outliers<br />2. Completely automated choices for trend and seasonal changes<br />3. Tried & tested method | 1. No prediction/confidence intervals<br />2. Ad hoc method with no underlying model<br />3. Only for quarterly & monthly data |
| X-12-ARIMA/<br />X-13-ARIMA                         |                                                              | 1. Allow adjustments for trading days and explanatory variables<br />2. Known outliers can be omitted<br />3. Level shifts & ramp effects can be modelled<br />4. Missing values estimated and replaced<br />5. Holiday factors can be estimated |                                                              |
| X-13-ARIMA-SEATS                                    |                                                              | 1. Model-based<br />2. Smooth trend estimate<br />3. Allows estimates at end-points<br />4. Incorporates changing seasonality |                                                              |
| STL<br />Seasonal & Trend Decomposition using Loess | - Iterative alogirthm<br />- Starts with $\hat T = 0$<br />- Uses mixture of loess and moving averages to successively refine trend & seasonal estimates<br />- Trend window controls loess bandwidth applied to de-seasonalized values<br />- Season window controls loess bandwidth applied to detrended subseries<br />- Seasonal component allowed to change over time; Rate of change controlled by analyst<br />- Smoothness of trend controlled by analyst | - Versatile<br />- Robust<br />- Handle any type of seasonality | - Only additive (Use log/Box-Cox transformations for other)<br />- No training day/calendar adjustments |

## Classical Decomposition

|                | $y_t$                       | Appropriate when Magnitude of seasonal fluctuations proportional to level of series |
| -------------- | --------------------------- | ------------------------------------------------------------ |
| Addititive     | $S_t + T_t + R_t$           | ❌                                                            |
| Multiplicative | $S_t \times T_t \times R_t$ | ✅                                                            |

Alternatively, use Box-Cox transformation, and then use additive decomposition. Logs turn multiplicative relationship into additive

$$
\begin{aligned}
y_t &= S_t \times T_t \times R_t \\ 
\implies
\ln y_t &= \ln S_t + \ln T_t + \ln R_t
\end{aligned}
$$

### Trend Estimation

Centered moving averages, to combat odd order

$$
\begin{aligned}
\hat T_t &= \dfrac{1}{2m} \left( \sum_{i = -(k+1)}^k y_{t+i} + \sum_{i = -k}^{k+1} y_{t+i} \right) \\
\text{where } k &= \dfrac{m-1}{2}
\end{aligned}
$$

| Order ($m$) | Curve             | Data Retention                  |
| ----------- | ----------------- | ------------------------------- |
| Larger      | Smoother, flatter | Less<br />(end points are lost) |
| Smaller     | Noisy             | More                            |

Moving average of the same length of a season/cycle removes its pattern

### Seasonal Adjusted Data

Component excluding the seasonal component

### Detrended Series

$$
\begin{aligned}
y_t - \hat T_t &= \hat S_t + \hat R_t \\
\frac{y_t}{\hat T_t} &= \hat S_t \times \hat R_t
\end{aligned}
$$

### Seasonal component

Average of de-trended series for that season. For eg, average of all values in Januaries

You can constraint the seasonal components such that

$$
\hat S_1 + \hat S_2 + \dots + \hat S_{n} = 0 \\
\hat S_1 \times \hat S_2 \times \dots \times \hat S_{n} = m
$$

### Remainder Component

$$
\begin{aligned}
\hat R_t &= y_t - (\hat T_t + \hat S_t) \\
\hat R_t &= \dfrac{y_t}{\hat T_t \hat S_t}
\end{aligned}
$$

