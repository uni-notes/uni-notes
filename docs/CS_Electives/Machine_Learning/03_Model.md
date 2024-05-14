# Model

$$
\hat y = \hat f(x) + u \\ 
\hat f(x) = E[y \vert x]
$$

where

| Denotation     | Term                                                    | Comment                                                      |
| -------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| $x$            | input; feature; predictor                               |                                                              |
| $y$            | output; target; response                                |                                                              |
| $\hat y$       | prediction                                              |                                                              |
| $E[y \vert x]$ | CEF (Conditional Expectation Function)                  |                                                              |
| $\hat f$       | Target function<br />Hypothesis<br />Model              | Gives mapping b/w $x$ and $y$ to obtain CEF                  |
| $p(y \vert x)$ | Target distribution/<br />Posterior distribution of $y$ | Gives mapping b/w $x$ and $y$ to obtain Conditional Distribution |
| $u$            | Random component                                        |                                                              |

## Desired Properties

- Unbiased: Mean of residuals = 0
- Efficient: Variance of residuals and learnt parameters is min
- Maximum likelihood $P(D, \theta)$
- Robust
- Consistent: $n \to \infty \implies E[u_i] \to 0$



Attributes of probabilistic forecast quality

1. Reliable: probabilistic calibration
   1. For quantile forecasts with level $\alpha$, observations $y_{t+k}$ should be less than $\hat y_{t+k}$ $\alpha$ times
   2. For interval forecasts with coverage $p$, observations $y_{t+k}$ should be within the interval $p$ times
   3. For predictive densities composed of $m+1$ quantile forecasts with nominal levels $\alpha_0, \alpha_1, \dots, \alpha_m$, all these quantile forecasts are evaluated individually using the above
   4. Q-Q Plots
2. Sharp: informative
   1. Concentration of probability: how tight the predictive densities are
   2. Perfect probabilistic forecast gives a probability of 100% on a single value
   3. CRPS
      1. Average of each predictive density and corresponding observation
      2. $\text{CRPS}_{t, h} = \int_y \ \Big( \hat F_{t+h \vert t} - 1(y_{t+h} \le y) \Big)^2 \ \cdot dy$
      3. $\text{CRPS}_h = \text{avg}(\text{CRPS}_{t, h})$
3. Skilled
4. High resolution

## Note

Every model is only limited to its ‘scope’, which should be clearly documented

## Model Types

|                                 | Ideal                                                        | Non-Parametric<br />(Nearest Neighbor)                       | Semi-Parametric               | Parametric                                                   |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| $\hat y$                        | $\text{Mean}(y \vert x)$                                     | $\text{Mean} \Big(y \vert x_i \in N(x) \Big)$<br />$N(x)$ is neighborhood of $x$ |                               | $f(x)$                                                       |
| Functional Form assumption      | None                                                         | None                                                         |                               | Assumes functional form with a finite & fixed number of parameters, before data is observed |
| Advantages                      | Perfect accuracy                                             | - learns complex patterns<br />- in a high-dimensional space<br />- without being specifically directed<br />- learns interactions |                               | Compression of model into a single function                  |
| Limitation                      | Not possible to obtain                                       | Suffers from curse of dimensionality: Requires large dataset, especially when $k$ is large<br /><br />Black box: Lacks interpretability<br />Large storage cost: Stores all training records<br />Computationally-expensive |                               | Lost information?                                            |
| Visualization                   | ![image-20240212222001445](./assets/image-20240212222001445.png) | ![image-20240212222208110](./assets/image-20240212222208110.png) |                               |                                                              |
| Space Complexity is function of |                                                              | Training set size                                            | Number of function parameters | Number of function parameters                                |
| Example                         |                                                              | Nearest Neighbor averaging                                   | Spline                        | Linear Regression                                            |

Fundamentally, a parametric model can be though of data compression

## Modelling Types

|                            | Discriminative/<br />Reduced Form                            | Generative/<br />Structural/<br />First-Principles           | Hybrid/<br />Discrepancy                   |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ |
| Characteristic             | Mathematical/Statistical                                     | Theoretical<br />(Scientific/Economic)                       | Mix of first principles & machine learning |
| Goal                       | 1. $\hat p(y \vert x)$<br />2. $\hat y = \hat E(y \vert x)$  | 1. $\hat p(x, y)$<br />2. $\hat p(y \vert x)$<br />3. $\hat y = \hat E(y \vert x)$ | $\hat y = \text{g}(x) + d(x)$              |
| Includes Causal Theory     | ❌                                                            | ✅                                                            | Same as Structural                         |
| Intrapolation?             | ✅                                                            | ✅                                                            | ✅                                          |
| Interpolation              | ⚠️                                                            | ✅                                                            | ⚠️                                          |
| Extrapolation              | ❌                                                            | ✅                                                            | ❌                                          |
| Counter-factual simulation | ❌                                                            | ✅                                                            | ❌                                          |
| Synthetic data generation  | ❌                                                            | ✅                                                            | ❌                                          |
| Out-of-Sample Accuracy     | Low                                                          | High<br />(only for good theoretical model)                  | Same as Structural                         |
| Limitations                | Unstable for equilibrium effects                             |                                                              |                                            |
| Derivation Time            | 0                                                            | High                                                         | Same as Structural                         |
| Example models             | Non-Probabilistic classifiers                                | Probabilistic classifiers (Bayesian/Gaussian)                |                                            |
| Example                    | 1. Mars position wrt Earth, assuming that Mars revolves around the Earth<br /><br />2. Relationship of wage vs college education directly | 1. Mars position wrt Earth, assuming that Mars & Earth revolve around the Sun<br /><br />2. Relationship of wage vs college education, with understanding of effects of supply of college educated students in the market |                                            |
| Example 2                  | Linear regression of wage wrt education                      | Wage & education using supply-demand curve                   |                                            |
| Comment                    | The shortcoming of reduced form was seen in the 2008 Recession<br/>The prediction model for defaults was only for the case that housing prices go up, as there was data only for that. Hence, the model was not good for when the prices started going down. | Learning $p(x, y)$ can help understand $p(u, v)$ if $\{x, y \}$ and $\{ u, v \}$ share a common underlying causal mechanism<br /><br />For eg: Apples falling down trees and the earth orbiting around the sun both inform us of the gravitational constant. |                                            |
| Example                    | $y = x^2, \hat y = x$<br />$y=e^x, \hat y=x^2$               | $y = x^2, \hat y = x^2$                                      |                                            |

![image-20240202174854454](./assets/image-20240202174854454.png)

### Structural vs Reduced-Form

![Structural model vs Reduced form](./assets/image-20240420134857116.png)

## Number of Variables

|          | Univariate Regression   | Multi-Variate                     |
| -------- | ----------------------- | --------------------------------- |
| $\hat y$ | $f(X_1)$                | $f(X_1, X_2, \dots, X_n)$         |
| Equation | $\beta_0 + \beta_1 X_1$ | $\sum\limits_{i=0}^n \beta_i X_i$ |
| Best Fit | Straight line           | Place                             |

## Degree of Model

|                         | Simple Linear Regression              | Polynomial Linear Regression                                 | Non-Linear Regression                                        |
| ----------------------- | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Equation                | $\sum\limits_{j=0}^k \beta_j X_j$     | $\sum \limits_{j=0}^k \sum\limits_{i=0}^n \beta_{ij} (X_j)^i$ | Any of the $\beta$ is not linear                             |
| Example                 | $\beta_0 + \beta_1 X_1 + \beta_1 X_2$ | $\beta_0 + \beta_1 X_1 + \beta_1 X_1^2 + \beta_1 X_2^{10}$   | $\beta_0 + e^{\textcolor{hotpink}{\beta_1} X_1}$             |
| Best Fit                | Straight line                         | Curve                                                        | Curve                                                        |
| Solving method possible | Closed-Form<br />Iterative            | Closed-Form<br />Iterative                                   | Iterative                                                    |
| Limitations             |                                       |                                                              | 1. Convergence may be slow<br />2. Convergence to local minima vs global minima<br />3. Solution may depend heavily on initial guess |
| Comment                 |                                       |                                                              | You can alternatively perform transformation to make your regression linear, but this isn’t best<br/>1. Your regression will minimize transformed errors, not your back-transformed errors (what actually matters). So the weights of errors will not be what is expected<br/>2. Transformed errors will be normal, but your back-transformed errors (what actually matters) won’t be a normal |

The term linear refers to the linearity in the coefficients $\beta$s, not the predictors

### Jensen’s Inequality

$$
E[\log y] < \log (E[y])
$$

Therefore
$$
\hat y = \exp(\beta_0 + \beta_1 x) + u_i \\
E[y \vert x] \ne E[\exp(\beta_0 + \beta_1 x)]
$$
However, if you assume that $u \sim N(0, \sigma^2)$
$$
E[y \vert x] = \exp(\beta_0 + \beta_1 x + \dfrac{\sigma^2}{2})
$$

## Categorial Inputs

### Binary

$$
\begin{aligned}
\hat y &= \beta_0 + \beta_1 x + \beta_2 T + + \beta_3 x T + u \\
\\
T = 0 \implies \hat y &= \beta_0 + \beta_1 x + u \\
T = 1 \implies \hat y &= (\beta_0 + \beta_2) + (\beta_1+\beta_3) x + u
\end{aligned}
$$

where $T=$ treatment/binary var

### Discrete Var

For $C$ possible values of discrete var, you need to create $(C-1)$ dummy vars, as all zeros is also scenario

Else, if you have $C$ dummy vars, you will have perfect multi-collinearity (dangerous)

