# Distribution Tests

## Normality

- Histogram with Kernel Density Estimation
- Q-Q Plots
- Moment tests

### Jaque-Bera Test

Tests for skewness and kurtosis combined
$$
\left[
\dfrac{\mu_3}{\text{SE}(\mu_3)}
\right]^2 + 
\left[
\dfrac{\mu_4}{\text{SE}(\mu_4')}
\right]^2
\sim
\chi^2_2
$$

### Shapiro-Wilk Test

$H_0:$ Sample $x$ comes from normal-distribution

Characteristics of test

- Defined for $n \ge 3$
- Best power for a given significance compared to other popular tests

Limitations

- This test is sample-size biased
  - Small sample size doesn't have enough information to conclude with high certainty
  - For a large dataset, even a small departure from normality will trigger a rejection
  - hence normal Q-Q plot should be used to confirm test results
- Failure to reject $H_0$, ie accepting $H_1$ is not proof that the distribution is normal
- Rejecting $H_0$ does not tell you how much the distribution differs from normal distribution

Test statistic

- $w \in (0, 1]$
- Very similar to correlation coefficient of a normal $Q-Q$ plot
- $w$ independent of location and scale of $x$

$$
w = \dfrac{(\sum a_i x_i)^2}{\sum (x_i - \bar x)^2}
$$

where

- $x_i=$ $i$th smallest value
- $a_i=$ Shapiro-Wilk Constant

### Note

- All tests are very sensitive to outliers
  - One outlier: distribution appears skewed
  - Two symmetric outliers: distribution appears to have heavy tails

### 