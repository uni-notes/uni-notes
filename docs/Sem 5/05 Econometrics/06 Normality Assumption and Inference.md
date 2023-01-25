## Probability Distribution of $\beta_2$

$$
\beta_2 =
\frac{}{}
$$

As $k_i$ is non-stochastic, the value of $\beta_2$ depends on $y$, which is a random variable

$$
\hat \beta_2 =
\sum k_i (\beta_1 + \beta_2 X_i + u_i)
$$
As $u_i$ is the only random component in the equation, $\beta_2 = f(u_i)$ meaning that we can assume $\beta_2$ follows the same distribution as $u_i$

As $u_i$ is normally distributed, $\beta_1$ and $beta_2$ are also normally-distributed

## Classical Linear Regression

Assumption of $u_i$

- Mean = 0
- Variance = $\sigma^2$
- Cov$(u_i, u_j) = 0$

$u_i \ N()$

$u_i$ and $u_j$ have [IID](#IID)

## Approximation for $t$ distribution

Student’s $t$ distribution

$$
\begin{align}
t
&= \frac{\hat \beta_2 - \beta_2}{SEE} \\&= \frac{(\hat \beta_2 - \beta_2) \sqrt{\sum x_i^2} }{\hat \sigma} \\
\end{align}
$$

This $t$ variable will follow $t$ distribution with $(n-2)$ degrees of freedom

## IID

Identical and Independent Distribution

## Central Limit Theorem

> The central limit theorem states that if you have a population with mean $\mu$ and standard deviation $\sigma$ and take sufficiently large random samples from the population with replacement, then the distribution of the sample means will be approximately normally distributed.
