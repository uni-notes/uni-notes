# Generalized Linear Model

## Condition

GLM will give similar performance to directly optimizing the ideal loss function only when $n$ is large

## Steps

1. Let $y$ have a probability distributions as long as it is from the exponential family
   - Included
     - Normal, log-normal, exponential, gamma, chi-squared, beta, Bernoulli, poisson, binomial, etc
   - Not included:
     - Student’s $t$ due to heavy tails
     - Mixed distributions (with different location/scale parameters)
2. Allow for any transformation (link function) of $y$, such that transformation is monodic and differentiable

3. Write linear parameters

4. Derive MLE

## Transformations

| Distribution             | Typical Uses                                       | Link Name             | Link Function<br />$x = g(y)$                 | Prediction Function<br>$\hat y$ |
| ------------------------ | -------------------------------------------------- | --------------------- | --------------------------------------------- | ------------------------------- |
| Normal/<br />Gaussian    | Linear response data                               | Identity              | $y$                                           | $\hat f$                        |
| Bernoulli/<br />Binomial | Outcome of single yes/no occurence                 | Logit<br />(Logistic) | $\ln \left \vert \dfrac{y}{1-y} \right \vert$ | $\sigma(\hat f)$                |
| Exponential/<br />Gamma  | Exponential response data<br>Scale parameters      | Negative Inverse      | $\dfrac{-1}{y}$                               | $\dfrac{-1}{\hat f}$            |
| Inverse Gaussian         |                                                    | Inverse Squared       | $\dfrac{1}{y^2}$                              | $\dfrac{1}{\sqrt{\hat f}}$      |
| Poisson                  | Count of occurrences in fixed amount of time/space | Log                   | $\ln \vert y \vert$                           | $e^{\hat f}$                    |
| Negative Binomial        | Poisson with varying variance                      |                       |                                               |                                 |
| Quasi                    | Normal with constant variance                      |                       |                                               |                                 |
| Quasi-binomial           | Binomial with constant variance                    |                       |                                               |                                 |
| Quasi-poisson            | Poisson with constant variance                     |                       |                                               |                                 |

Better to fit for Gamma than Poisson
- Hard to determine optimal time interval for Poisson
	- In scenarios where events occur in bursts or have high variability, the Poisson distribution may not adequately capture this overdispersion
	- if many occurrences happen in a short time frame, using a large time interval with Poisson will result in this going unnoticed
	- Gamma would flag these anomalies due to its flexibility in handling varying rates
- Hard to determine optimal phase of interval for Poisson
	- Starting at 00:00 vs 00:05 will give different results
- You can invert the obtain $\lambda$ from Gamma and get the poisson distribution

## Uncertainty

Generalized linear model: https://fromthebottomoftheheap.net/2018/12/10/confidence-intervals-for-glms/

Exponential regression confidence intervals will use similar logic

Assume that $f$ inside $e^f$ is t-distributed with
- mean 0
- std = model rmse (in the link space)
- dof = $n-k$

As this is time series,

$$
\begin{aligned}
var(f_t+h) &= \sum_{i=1}^h var(f_t+i) \\
var(f_t+h) &= h*var(f_t)
\end{aligned}
$$

Compute statistics **independently** in the link scale
- Standard deviation
- Quantiles
- Confidence intervals

Finally, back-transform each statistic **independently** to response scale

#### Example

- $y = a e^x$
	- $Q(y, q) = a e^{Q(x, q)}$