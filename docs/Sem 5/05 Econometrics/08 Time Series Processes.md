## Time Series

Observation of random variable ordered by time

Time series variable can be

- Time series at level (absolute value)
- Difference series (relative value)
    - First order difference $\Delta y_t = y_t - y_{t-1}$
    - Called as ‘returns’ in finance
    - Second order difference $(\Delta y_t)_2 = \Delta y_t - \Delta y_{t-1}$

## Univariate Time Series

Basic model only using a variable’s own properties like lagged values, trend, seasonality, etc

## Why do we use different techniques for time series?

This is due to

- behavioral effect
- history/memory effect
    - Medical industry always looks at the records of your medical history
- Inertia of change

## Components of Time Series Processes

### Auto-correlation

High possibility of auto-correlation

Sometimes just auto-correlation is enough to learn the values of a value

$$
y_t = \beta_0 + \beta_1 y_{t-1} + e_t

$$
If we take $j$ lags,

$$
y_t = \beta_0 + \sum_{i=1}^j \beta_i y_{t-i} + e_t

$$
Generally, $i>j \implies \beta_i < \beta_j$

Impact of earlier lags is lower than impact of recent lags

### Shock

‘Shock’ is an abrupt/unexpected deviation(inc/dec) of the value of a variable from its expected value

This incorporates influence of previous disturbance

They cause a structural change in our model equation. Hence, we need to incorporate their effect.

$$
\text{Shock}_t = y_t - E(y_t)
$$

Basically, shock is basically $u_t$ but it is fancily called as a shock, because they are large $u$

Can be

|          | Temporary                                                    | Permanent                                                    |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Duration | Short-term                                                   | Long-Term                                                    |
|          |                                                              | Causes structural change                                     |
| Examples | Change in financial activity due to Covid                    | Change in financial activity due to 2008 Financial Crisis    |
|          | Heart rate change due to minor stroke<br />Heart rate change due to playing | Heart rate change due to major stroke<br />Heart rate change due to big injury |
|          | Goals scored change due to small fever                       | Goals scored change due to big injury                        |

Model becomes

$$
y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 u_{t-1}
$$

### Structural Breaks

Permanent change in the variable causes permanent change in relationship

We can either use

- different models before and after structural break
- binary ‘structural dummy variable’ to capture this effect

For eg, long-term injury

$$
y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 B_t
$$

### Trend

Tendency of time series to change at a certain expected rate.

Trend can be

- deterministic/systematic (measurable)
- ==random/stochastic (not measurable $\implies$ cannot be incorporated)==

For eg: as age increases, humans have a trend of

- growing at a certain till the age of 20 or so
- reducing heart rate

Model becomes

$$
y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 t
$$

### Seasonality/Periodicity

Tendency of a variable to change in a certain manner at regular intervals.

We use binary ‘seasonal dummy’ variables to capture this effect

For eg

- demand for woolen clothes is high every winter
- demand for ice cream is high every summer

Finance industry has ‘anomalies’

$$
y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 S_t
$$

### Volatility

Measure of variation of a variable from its expected value

If the variance is heteroschedastic (changes over time), the variable is volatile

$$
\begin{align}
\sigma^2_{y_t}
&= E \Big [\Big(y_t - E(y_t) \Big)^2 \Big] \\&= E \Big [\Big(y_t - \textcolor{hotpink}{0} \Big)^2 \Big] \quad \text{(if \textcolor{hotpink}{normalized})} \\&= E [y_t^2 ] \\&= y_t^2 \\
\end{align}
$$

$$
y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 \sigma^2_{t-1}
$$

## Volatility Models

### ARCH

AutoRegressive Conditional Heteroschedacity models

1. Calculate $y_t = f(y_{t-1}, u_t)$

2. Calculate $\hat u_t$

3. $$
   y_t = u_t \implies \sigma^2 (y_t) = \sigma^2 (u_t)
   
$$

4. So our model becomes
   $$
   (u_t)^2 = \lambda_0 + \lambda_1 (u_{t-1})^2 + \dots
   
$$

5. 
### GARCH

Generalized AutoRegressive Conditional Heteroschedacity models

## Lag Terms

$$
\begin{align}
\text{Let's say} \to
y_t &= f(y_{t-2}) \\y_t &= f(y_{t-1}) \\y_{t-1} &= f(y_{t-2})
\end{align}
$$

$$
y_t = \rho_1 y_{t-1} + \rho_2 y_{t-2} + u_t
$$

Here, $\rho_1$ and $\rho_2$ are partial-autocorrelation coefficient of $y_{t-1}$ and $y_{t-2}$ on $y_t$

$$
y_t = \rho_1 y_{t-2} + u_t

$$
Here, $\rho_1$ is total autocorrelation coefficient of $y_{t-2}$ on $y_t$

We choose the number of lags by trial-and-error and checking which coefficients are significant ($\ne 0$)

## Data-Generating Process

### White Noise Series

$y_t$ whose lags have no impact on current value ($y_t$ is independent of lags), ie, both partial autocorrelation coefficient **and** total autocorrelation coefficient associated with each lag is statistically 0.

Consider a distribution of $y$ for each time period has

- 0 mean
- 0 correlation between $x$ and $y$
- Identical variance

$$
\begin{align}
\text{Consider } y_t
&= \sum \rho_i y_{t-i} + u_t \\&= \sum \textcolor{hotpink} 0 \times y_{t-i} + u_t
\quad \text{(Independent of lags)} \\
\implies y_t &= u_t \\
& \text{and} \\
E(y_t) &= \mu = 0 \\E[(y_t - \mu)^2] &= \sigma^2
\end{align}
$$

If a financial series is a white noise series, then we say that the ‘market is efficient’

### Stationary Stochastic

Earlier past is less important compared to recent past. Less susceptible to permanent shock

$$
y_t = \beta_1 y_{t-1} + u_t \\
0 < |\beta_1| < 1
$$

This is basically a wave (for eg, sound-wave), and $\beta$ is basically the amplitude of the wave

- $0 < |\beta_1| < 1$
    - Series oscillates
    - Series has Mean-reverting tendancy
    - Otherwise if
    - $|\beta_1| > 1$, it’ll be explosive (linearly-increasing/decreasing)
      - This is theoretically-possible, but not possible in real-world
- Mean, variance and autocovariance are time-invariant
  (Mean, variance of distribution of possible outcomes, corresponding to each time period is same)
    - Distribution of the variable remains constant for each time instant ==(not across time periods)==
    - Basically this series have homoscedascity of time-series variable
    - $E(y_t) = \mu$ can be zero/non-zero
    - $E[(y_t - \mu)^2] = \sigma^2$
    - $E[(y_t - \mu)(y_{t+k}-\mu)] = r_k$
    - Hence, in case of any shocks, the series returns to the original
  

#### Example

- GDP Growth rate
  GDP keeps changing (mostly increasing), but the rate of growth remains constant
- Interest rate

#### Derivation of properties

$$
\begin{align}
y_t &= \beta^t y_0 + \sum_{i=0}^{t-1} \beta^i u_{t-1} + u_t \\
\implies
E(y_t) &= \beta^t y_0 \to \text{Constant} \\
\implies
E[y_t-\beta^t y_0] &= \sum_{i=0}^t \beta^t \sigma^2 \\&= \alpha(1 + \beta + \beta^2 + \dots) \\&= \frac{\sigma^2}{1 - \beta} \to \text{Constant}
\end{align}
$$

### Non-Stationary Stochastic

Will have either ==**one/both**== of the following

- Mean at each time period is ==**different**== across all time periods
    - Mean of distribution of possible outcomes corresponding to each time period is different

- Variance at each time period is ==**different**== across all time periods
    - Variance of distribution of possible outcomes corresponding to each time period is different

We need to tranform this somehow, as OLS and [GMM](#GMM) cannot be used for non-stationary processes, because the properties of OLS are violated - heteroscedastic variance of error term
#### Random Walk w/o drift (Pure random walk process)

- Mean is constant over time
- Variance varies over time

It is a long memory series

$$
\begin{align}
y_t
&= \beta_1 y_{t-1} + u_t \\&= 1 y_{t-1} + u_t & (\beta_1 = 1) \\&= y_{t-1} + u_t \\
\implies
y_t &= y_0 + \sum_{i=0}^t u_i
\end{align}
$$

Every value is basically the

- Previous value $\pm$ something
- Initial value + cumulative sum of disturbances

$$
\begin{align}
E(y_t)
&= E(y_0) + E(u_1) + E(u_2) + \dots + E(u_t) \\&= y_0 + 0 \\
\implies E(y_t) &= y_0
\end{align}
$$

$$
E[(y_t - y_0)^2] = t \sigma^2
$$

Volatility increases over time

#### Random Walk w/ drift

- Mean varies over time
- Variance varies over time

It is a [Long memory series](#Long memory series)

Similar to [Random Walk w/o drift](#Random Walk w/o drift), but has  $\beta_0$ as well

$$
\begin{align}
y_t
&= \beta_0 + y_{t-1} + u_t \\&= t\beta_0 + y_0 + \sum_{i=0}^t u_i
\end{align}
$$

$$
E(y_t) = t \beta_0 + y_0
$$

$$
E[(y_t - t\beta_0 - y_0)^2] = t \sigma^2
$$

Volatility increases over time

#### Random Walk w/ drift and deterministic trend

- Mean changes over time
- Variance changes over time

$$
\begin{align}
y_t
&= \beta_0 + \beta_1 t + y_{t-1} + u_t

& (\beta_0 \ne 0, \beta_1 \ne 0) \\
&=
y_0 + t \beta_0 + \beta_1 \sum_{i=1}^t i + \sum_{i=1}^t u_t 
\end{align}
$$

$\beta_1$ is deterministic = stochastic = non-random

$$
E(y_t) = t \beta_0 + \beta_1 \sum_{i=1}^t i + y_0
$$

$$
E\Big[
(y_t = t \beta_0 - \beta_1 \sum_{i=1}^t i - y_0 )^2
\Big]
= t \sigma^2
$$

#### Random Walk w/ drift and non-deterministic trend

$\beta_1$ is non-deterministic = non-stochastic = random

We cannot predict this easily

## Integrated/DS Process

Difference Stationary Process

A non-stationary series is said to be integrated of order $k$, if mean and variance of $k^\text{th}$-difference are time-invariant

If the first-difference is non-stationary, we take second-difference, and so on

### Pure random walk is DS

$$
\begin{align}
y_t &= y_{t-1} + u_t \\
\implies \Delta y_t &= \Delta y_{t-1} + u_t
\quad \text{(White Noise Process = Stationary)}
\end{align}
$$

### Random walk w/ drift is DS

$$
\begin{align}
y_t &= \beta_0 + y_{t-1} + u_t \\
\implies \Delta y_t &= \beta_0 + \Delta y_{t-1} + u_t
\quad \text{(Stationary)}
\end{align}
$$

## TS Process

Trend Stationary Process

A non-stationary series is said to be …, if mean and variance of de-trended series are time-invariant

Assume a process is given by

$$
y_t = \beta_0 + \beta_1 t + y_{t-1} + u_t

$$
where trend is deterministic/stochastic

Then

- Time-varying mean
- ==Constant variance ???==

We perform **de-trending** $\implies$ subtract $(\beta_0 + \beta_1 t)$ from $y_t$

$$
(y_t - \beta_0 - \beta_1 t) = y_{t-1} + u_t

$$
If

- $\beta_2 = 0$, the de-trended series is white noise process
- $\beta_2 \ne 0$, the de-trended series is a stationary process

**Note**
Let’s say $y_t = f(x_t)$

If both $x_t$ and $y_t$ have equal trends, then no need to de-trend, as both the trends will cancel each other

## Unit Root Test for Process Identification

$$
y_t = \textcolor{hotpink}{\beta_1} y_{t-1} + u_t
$$

| $\textcolor{hotpink}{\beta_1}$ | $\gamma$ | Process        |
| ------------------------------ | :------: | -------------- |
| $0$                            |          | White Noise    |
| $(0, 1)$                       |          | Stationary     |
| $[1, \infty)$                  |          | Non-Stationary |

### Augmented Dicky-Fuller Test

- $H_0: \beta_1=1$
- $H_0: \beta_1 \ne 1$

Alternatively, subtract $y_{t-1}$ on both sides of main equation

$$
\begin{align}
y_t - y_{t-1} &= \beta_1 y_{t-1} - y_{t-1} + u_t \\y_t - y_{t-1} &= (\beta_1-1) y_{t-1} + u_t \\
\Delta y_t &= \gamma y_{t-1} + u_t & (\gamma = \beta_1 - 1)
\end{align}
$$

- $H_0: \gamma=1$ (Non-Stationary)
- $H_1: \gamma \ne 1$ (Stationary)

If p value $\le 0.05$

- we reject null hypothesis and accept alternate hypothesis
- Hence, process is stationary

We test the hypothesis using Dicky-Fuller distribution, to generate the critical region

| Model           | Hypotheses $H_0$ | Test Statistic |
| --------------- | ---------------- | -------------- |
| $\Delta y_t = $ |                  |                |
|                 |                  |                |
|                 |                  |                |

## Long memory series

Earlier past is as important as recent past

## Q Statistic

Test statistic like $z$ and $t$ distribution, which is used to test ‘joint hypothesis’

## GMM

Generalized method of moments