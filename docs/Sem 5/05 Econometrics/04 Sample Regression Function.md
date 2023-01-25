## Sample

Subset of Population

Useful, as we can never have access to the entire population

We will only have limited values of $y$ for given value(s) of $x$; we don’t have access to the true distribution of $y$ for different values of $x$

## SRF

Sample Regression Function

$$
\begin{align}
\hat y_i
&= \hat \beta_0 + \hat \beta_1 x_i \\
y_i
&= \hat \beta_0 + \hat \beta_1 x_i + \hat u_i
\end{align}
$$

We try to make each of these hyperparameters close to their PRF counter-parts

- $\hat \beta_0$ is an estimator of $\beta_0$ from the PRF
- $\hat \beta_1$ is an estimator of $\beta_1$ from the PRF
- $\hat u_i$ is an approximation of $u_i$ from the PRF
- $\hat y_i$ is the predicted value of dependent variable from sample data
- $y_i$ is the true value of dependent variable

We can use the SRF to approximate a variable’s distribution, by using statistical distributions, such as Poisson, $t$, Normal, $\chi^2$

### Problem

- Over-estimation
  SRF estimate may be higher than the PRF estimate
- Under-estimation
  SRF estimate may be lower than the PRF estimate

## Issue with Sample Data

We will get different SRF for each sample

Perfect fit is impossible, due to **Sample Fluctuations**

- The tendancy of each sample to be different from each other
- It is basically impossible to avoid this

There is no way to say what best represents the PRF

Sample can be **mis-representing**
You have to be careful about interpreting the results

- Nature of random sample may cause SRF to over-estimate/under-estimate
- Biased choosing of sample
  - Researcher chooses a particular sample
  - For eg
    - I chose Ronaldo for 2nd year study project
    - Choosing students of high attendance only

## Sampling Distribution

The distribution of $\hat \beta_0, \hat \beta_1, \hat u_i$ for different samples

### Why is it important?

Helps us understand which pair of $\hat \beta_0, \hat \beta_1$ is the closest to the PRF $\beta_0, \beta_1$

## Techniques to identify SRF

You don’t have to memorize the formula, but you should know what is happening; that way you can debug any errors

- OLS
- Maximum Likelihood Estimation

## Autocorrelation

Correlation between values of the same variable

Usually used for time series data

We use ARIMA(AutoRegressive Integrated Moving Averages) Model
It’s just values based on lagged values
