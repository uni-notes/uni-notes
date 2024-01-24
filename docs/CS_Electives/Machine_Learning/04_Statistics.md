# Statistics

Statistical concepts such as Parameter estimation, Bias, Variance help in the aspects of generalization, over-fitting and under-fitting

## Estimation Types

| Estimation Type | Regression Output                         | Classification Output                                        |
| --------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Point           | $E(y \vert X)$                            | $E(c_i \vert X)$                                             |
| Probabilistic   | $E(y \vert X)$<br />$\sigma^2(y \vert X)$ | $P(c_i \vert X)$<br />This is not the model confidence! This is the likelihood |

## Likelihood vs Confidence

- Likelihood is the probability of classification being of a certain class
  - Unreliable if input is unlike anything from training
- Confidence is the model’s confidence that the likelihood is correct

## IDK

|          | Expected deviation              |
| -------- | ------------------------------- |
| Bias     | from the true value             |
| Variance | caused by any particular sample |

## Function Estimation

Estimation of relationship b/w input & target variables, ie predict a variable $y$ given input $x$

$$
y = \hat y + \epsilon
$$

where $\epsilon$ is Bayes Error

## Statistical Learning Theory

Helps understand performance when we observe only the training set, through assumptions about training and test sets

- i.i.d (independent & identical distributions)

  - Training & test data arise from same process
  - Observations in each data set are independent
  - Training set and testing set are identically distributed

- $X$ values are fixed in repeated sampling

- No specification bias

   - We need to use the correct functional form, which is theoretically consistent

- No Unbiasedness

   - Independent vars should not be correlated with each other
   - If |correlation| > 0.5 between 2 independent vars, then we drop one of the variables

- High DOF

   - Degree of freedom $= n - k$, where
     - $n =$ number of observations
     - $k =$ no of independent variables
   - DOF $\to$ 0 leads to overfitting

- High coefficient of variation in $X$

   - We need more variation in values of $X$
   - Indian stock market is very volatile. But not in UAE; so it's hard to use it an independent var. Similarly, we cant use exchange rate in UAE as a regressor, as it is fixed to US dollars

- Series of $u_i$ are random

   1. $E(u_i | x_i) = 0$

     2. Symmetric distribution for values of error terms **for a given value $x$**
     3. **Not** over time/different values of $x$
     4. This means that
        1. you have used up all the possible factors
        2. $u_i$ only contains the non-systematic component

   5. Homoscedascity of variance

     6. $\text{var}(u_i | x_i) = \sigma^2 (u_i|x_i) = \text{constant}$ should be same $\forall i$
     7. For the Variance of distribution of potential outcomes, the range of distribution stays same over time
     8. $\sigma^2 (x) = \sigma^2(x-\bar x)$

     else, the variable is **volatile**; hard to predict; **we cannot use OLS**

     - if variance decreases, value of $y$ is more reliable as training data
     - if variance increases, value of $y$ is less reliable as training data
     - We use voltaility modelling (calculating variance) to predict the pattern in variance

- No covariance/correlation between $u_i$ and $x_i$: $\text{Cov}(u_i, x_i) = 0$

   - No relationship between error and independent variables
   - If there is correlation, then we cannot correctly obtain coefficients

- No [autocorrelation](#Autocorrelation) between $u_i$ and $u_j$: $\text{cov}[ (u_i, u_j) | (x_i, x_j) ]=0$

   - Residual series should be independent of other residual series

   - For any 2 values $x_i$ and $x_j$, the correlation between $u_i$ and $u_j$ is $0$

      - If we plot the scatter plot between $u_i$ and $u_j$, there should be no sign of correlation

- There is no measurement error $\delta_i$ in $X$ or $Y$

   - $X_\text{measured} = X_\text{true}$
   - $y_\text{measured} = y_\text{true}$
   - $E(\delta_i)=0$
   - $\text{var}(\delta_i | x_i) = \sigma^2 (\delta_i|x_i) = \text{constant}$ should be same $\forall i$
   - $\text{Cov}(\delta_i, x_i) = 0, \text{Cov}(\delta_i, u_i) = 0$

   If there is measurement error, we need to perform [correction](#Errors-in-Measurement Correction)

- If there exists autocorrelation in time series, then we have to incorporate the lagged value of the dependent var as an explanatory var of itself

But rarely used in practice with deep learning, as

- bounds are loose
- difficult to determine capacity of deep learning algorithms

## Attenuation Bias

High measurement error $\delta$ and random noise $u$ causes our estimated coefficients to be lower than the true coefficient
$$
\begin{aligned}
\lim_{n \to \infty} \hat \beta &= \beta \times \text{SNR} \\
\text{Signal-Noise Ratio: SNR} &= \dfrac{\sigma^2_x}{\sigma^2_x \textcolor{hotpink}{+ \sigma^2_u + \sigma^2_\delta}}
\end{aligned}
$$

## Errors-in-Measurement Correction

This can be applied to

- any learning algorithm
- for regressors or response variables(s)

Let’s say true values of a regressor variable $X_1$ was measured as $X_1^*$ with measurement error $\delta_1$, where $\delta_1 \ne N(0, 1)$. Here, we cannot ignore the error.

### Step 1: Measurement Error

Use an appropriate distribution to model the measurement error. Not necessary that the error is random.

For eg, if we assume that the error is a skewed normal-distributed with variance $\sigma^2_{X_1}$ signifying the uncertainty.

$$
\delta_1 = N(\mu_{X_1}, \sigma^2_{X_1}, \text{Skew}_{X_1}, \text{Kurt}_{X_1})
$$

### Step 2: Measurement

Model the relationship between the error and the measured value.

For eg, If we assume that the error is additive

$$
\begin{aligned}
X_1^* &= X_1 + \delta_1 \\
\implies X_1 &= X_1^* \textcolor{hotpink}{- \delta_1}
\end{aligned}
$$

### Step 3: Model

Since $X_1^*$ is what we have, but we want the mapping with $X_1$,

$$
\begin{aligned}
\hat y &= f(X_1) \\
&= f(X_1^* \textcolor{hotpink}{- \delta_1})
\end{aligned}
$$

### Example

Example: Modelling with linear regression using a regressor with measurement error
$$
\begin{aligned}
\implies \hat y
&= \theta_0 + \theta_1 X_1 \\
&= \theta_0 + \theta_1 (X_1^* - \delta_1) \\
&= \theta_0 + \theta_1 X_1^* \textcolor{hotpink}{- \theta_1 \delta_1}
\end{aligned}
$$

