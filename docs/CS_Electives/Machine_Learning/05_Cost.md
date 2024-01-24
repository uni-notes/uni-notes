# Cost Function

## Error

Usually, we define error as

$$
u_i = \hat y_i - y_i
$$

Bayes’ Error is the error incurred by an ideal model, which is one that makes predictions from true distribution $P(x,y)$; even such a model incurs some error due to noise/overlap in the distributions

### Deming Regression/Total Least Squares

Useful for when data has noise due to

- Measurement error
- Need for privacy etc, such as when conducting a salary survey.

$$
\begin{aligned}
u_i &= (\hat y_i - y_i)^2  + \lambda (\hat x_i - x_i)^2 \\
\hat y_i &= \hat \beta_0 + \hat \beta_1 x_i \\
\hat x_i &= \dfrac{y_i - \beta_0}{\beta_1} \\
\lambda &= \dfrac{\sigma^2(\text{known measurement error}_x)}{\sigma^2(\text{known measurement error}_y)}
\end{aligned}
$$

| Measurement Error of Regressor | $\lambda$ |            |
| ------------------------------ | --------- | ---------- |
| 0                              | 0         | OLS        |
| Same as Response               | 1         | Orthogonal |

![image-20231218180609992](./assets/image-20231218180609992.png)

![image-20231218180618882](./assets/image-20231218180618882.png)

## Loss Functions $L(\theta)$

$$
\text{Loss}_i = L(\theta, u_i)
$$

- Penalty for a single point (absolute value, squared)
- Only error terms

## Regression Loss

![image-20231218182207191](./assets/image-20231218182207191.png)

| Metric                                                  |                         $J(\theta)$                          |                       Preferred Value                        |                        Unit                        |        Range        | Signifies                                                    | Advantages<br />✅                                            | Disadvantages<br />❌                                         | Comment                                                      | $\alpha$ of advanced family |
| :------------------------------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------: | :-----------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | --------------------------- |
| BE<br />(Bias Error)                                    |                            $u_i$                             | $\begin{cases} 0, & \text{Unbiased} \\ >0, & \text{Over-prediction} \\ <0, & \text{Under-pred} \end{cases}$ |                    Unit of $y$                     | $(-\infty, \infty)$ | Direction of error bias<br />Tendency to overestimate/underestimate |                                                              | Cannot evaluate accuracy, as equal and opposite errors will cancel each other, which may lead to non-optimal model having error=0 |                                                              |                             |
| AE<br />(Absolute Error)/<br />L1                       |                     $\vert  u_i  \vert$                      |                         $\downarrow$                         |                    Unit of $y$                     |    $[0, \infty)$    |                                                              | Robust to outliers                                           | Does not penalize large deviations<br />Not differentiable at origin, which causes problems for some optimization algo |                                                              |                             |
| SE<br />(Squared Error)/<br />L2                        |                          ${u_i}^2$                           |                         $\downarrow$                         |                   Unit of $y^2$                    |    $[0, \infty)$    | Variance of errors with mean as MDE<br />Maximum log likelihood | Penalizes large deviations                                   | Sensitive to outliers                                        |                                                              | $\approx 2$                 |
| L3/<br />Smooth L1/<br />Pseudo-Huber/<br />Charbonnier | $\begin{cases} \dfrac{u_i^2}{2 \epsilon}, & \vert u_i \vert < \epsilon \\ \vert u_i \vert - \dfrac{\epsilon}{2}, & \text{otherwise} \end{cases}$ |                         $\downarrow$                         |                                                    |    $[0, \infty)$    |                                                              | Balance b/w L1 & L2<br />Prevents exploding gradients<br />Robust to outliers |                                                              |                                                              | 1                           |
| Huber                                                   | $\begin{cases} \dfrac{u_i^2}{2}, & \vert u_i \vert < \epsilon \\ \epsilon \vert u_i \vert - \dfrac{\epsilon^2}{2}, & \text{otherwise} \end{cases} \\ \epsilon_\text{recommended} = 1.345 \sigma_u$ |                         $\downarrow$                         |                                                    |    $[0, \infty)$    |                                                              | Same as Smooth L1                                            | Computationally-expensive for large datasets<br />$\epsilon$ needs to be optimized<br />Only first-derivative is defined | $H = \epsilon \times {L_1}_\text{smooth}$                    |                             |
| Log Cosh                                                | $\Big\{ \log \big( \ \cosh (u_i) \ \big) \Big \} \\ \approx \begin{cases} \dfrac{{u_i}^2}{2}, & \vert u_i \vert \to 0 \\ \vert u_i \vert - \log 2, & \vert u_i \vert \to \infty \end{cases}$ |                         $\downarrow$                         |                                                    |                     |                                                              | Same as Smooth L1<br />Doesn’t require hyperparameter tuning<br />Double differentiable |                                                              |                                                              |                             |
| Cauchy/<br />Lorentzian                                 | $\dfrac{\epsilon^2}{2} \times \log \Big[ 1 + \left( \dfrac{{u_i}}{\epsilon} \right)^2 \Big]$ |                         $\downarrow$                         |                                                    |                     |                                                              | Same as Smooth L1                                            | Not convex                                                   |                                                              | 0                           |
| Log-Barrier                                             | $\begin{cases} - \epsilon^2 \times \log \Big(1- \left(\dfrac{u_i}{\epsilon} \right)^2 \Big) , & \vert u_i \vert \le \epsilon \\ \infty, & \text{otherwise} \end{cases}$ |                                                              |                                                    |                     |                                                              |                                                              |                                                              | Regression loss $< \epsilon$, and classification loss further |                             |
| $\epsilon$-insensitive                                  | $\begin{cases} 0, & \vert u_i \vert \le \epsilon \\ \vert u_i \vert - \epsilon, & \text{otherwise} \end{cases}$ |                                                              |                                                    |                     |                                                              |                                                              | Non-differentiable                                           |                                                              |                             |
| Bisquare/<br />Welsch/<br />Leclerc                     | $\begin{cases} \dfrac{\epsilon^2}{6 \left(1- \left[1-\left( \dfrac{u_i}{\epsilon} \right)^2 \right]^3 \right)}, & \vert u_i \vert < \epsilon \\ \dfrac{\epsilon^2}{6}, & \text{otherwise} \end{cases} \\ \epsilon_\text{recommended} = 4.685 \sigma_u$ |                         $\downarrow$                         |                                                    |                     |                                                              | Robust to outliers                                           | Suffers from local minima (Use huber loss output as initial guess) | Ignores values after a certain threshold                     | $\infty$                    |
| Geman-Mclure                                            |                                                              |                                                              |                                                    |                     |                                                              |                                                              |                                                              |                                                              | -2                          |
| Quantile                                                | $\begin{cases} \sum q \vert u_i \vert , & \hat y_i < y_i \\ \sum (1-q) \vert u_i \vert, & \text{otherwise} \end{cases}$<br />$q = \text{Quantile}$ |                         $\downarrow$                         |                    Unit of $y$                     |                     |                                                              | Robust to outliers                                           | Computationally-expensive                                    |                                                              |                             |
| Expectile                                               | $\begin{cases} \sum e (u_i)^2 , & \hat y_i < y_i \\ \sum (1-e) (u_i)^2, & \text{otherwise} \end{cases}$<br />$e = \text{Expectile}$ |                         $\downarrow$                         |                   Unit of $y^2$                    |                     |                                                              |                                                              |                                                              | Expectiles are a generalization of the expectation in the same way as quantiles are a generalization of the median |                             |
| APE<br />(Absolute Percentage Error)                    |       $\left \lvert  \dfrac{ u_i }{y_i}  \right \vert$       |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    |                                                              | Easy to understand<br />Robust to outliers                   | Explodes when $y_i \approx 0$<br />Division by 0 error when $y_i=0$ |                                                              |                             |
| ASE<br />(Absolute Scaled Error)                        |     $\dfrac{ \lvert u_i \rvert }{\text{AE}(\bar y, y)}$      |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    |                                                              |                                                              |                                                              |                                                              |                             |
| ASE Modified                                            |  $\left \lvert \dfrac{ u_i }{\bar y_i - y_i} \right \vert$   |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    |                                                              |                                                              | Explodes when $\bar y - y_i \approx 0$<br />Division by 0 error when $\bar y - y_i \approx 0$ |                                                              |                             |
| WMAPE<br />(Weighted Mean Absolute Percentage Error)    | $\dfrac{1}{n} \left(\dfrac{ \sum \vert  u_i \vert  }{\sum \vert  y_i  \vert}\right)$ |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    |                                                              | Avoids the  limitations of MAPE                              | Not as easy to interpret                                     |                                                              |                             |
| MSLE<br />(Mean Squared Log Error)                      | $\dfrac{1}{n} \sum( \log_{1p} \hat y_i - \log_{1p} y_i  )^2$ |                         $\downarrow$                         |                   Unit of $y^2$                    |    $[0, \infty)$    | Equivalent of log-transformation before fitting              | - Robust to outliers<br />- Robust to skewness in response distribution<br />- Linearizes relationship | - Penalizes under-prediction more than over-prediction - Penalizes large errors very little, even lesser than  MAE (still larger than small errors, but weight penalty inc very little with error)<br />- Less interpretability |                                                              |                             |
| RMSLE<br />(Root MSLE)                                  |                     $\sqrt{\text{MSLE}}$                     |                         Same as MSE                          |                    Unit of $y$                     |     Same as MSE     |                                                              | Same as MSLE                                                 | Same as MSLE                                                 |                                                              |                             |
| RAE<br />(Relative Absolute Error)                      | $\dfrac{\sum \vert  u_i \vert}{\sum \vert  \bar y - y_i  \vert}$ |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    | Scaled MAE                                                   |                                                              |                                                              |                                                              |                             |
| RSE<br />(Relative Square Error)                        |       $\dfrac{\sum  (u_i)^2 }{\sum (\bar y - y_i)^2 }$       |                         $\downarrow$                         |                         %                          |    $[0, \infty)$    | Scaled MSE                                                   |                                                              |                                                              |                                                              |                             |
| Peer Loss                                               | $L(\hat y_i \vert x_i, y_i) - L \Big(y_{\text{rand}_j} \vert x_i, y_{\text{rand}_j} \Big)$<br />$L(\hat y_i \vert x_i, y_i) - L \Big(\hat y_j \vert x_k, y_j \Big)$<br />Compare the loss of actual prediction with respect to predicting a random value |                                                              |                                                    |                     | Actual information gain                                      | Penalize overfitting to noise                                |                                                              |                                                              |                             |
| Winkler score $W_{p, t}$                                |    $\dfrac{Q_{\alpha/2, t} + Q_{1-\alpha/2, t}}{\alpha}$     |                         $\downarrow$                         |                                                    |                     |                                                              |                                                              |                                                              |                                                              |                             |
| CRPS<br />(Continuous Ranked Probability Scores)        |               $\overline Q_{p, t}, \forall p$                |                         $\downarrow$                         |                                                    |                     |                                                              |                                                              |                                                              |                                                              |                             |
| CRPS_SS<br />(Skill Scores)                             | $\dfrac{\text{CRPS}_\text{Naive} - \text{CRPS}_\text{Method}}{\text{CRPS}_\text{Naive}}$ |                         $\downarrow$                         |                                                    |                     |                                                              |                                                              |                                                              |                                                              |                             |
| AIC<br />Akaike Information Criterion                   |   $-2 \log L + 2 (k+2)$,<br />where $L$ is the likelihood    |                         $\downarrow$                         | Equivalent to Leave-one-out cross validation score |                     |                                                              | Penalizes predictors more heavily than $R_\text{adj}^2$      | For small values of $n$, selects too many predictors         |                                                              |                             |
| AIC Corrected                                           |          $\text{AIC} + \dfrac{2(k+2)(k+3)}{n-k-3}$           |                         $\downarrow$                         |                                                    |                     |                                                              |                                                              |                                                              |                                                              |                             |
| BIC/SBIC/SC<br />(Bayesian Information Criterion)       | $-2 \log L + (k+2) \log n$,<br />where $L$ is the likelihood |                         $\downarrow$                         |                                                    |                     |                                                              | Penalizes predictors more heavily than AIC                   |                                                              | Minimizing equivalent to minimizing leave-one-out cross validation score when $v = n \left[ 1 - \dfrac{1}{\log \vert n \vert -1} \right]$ |                             |

### Outlier Sensitivity

![image-20231220103800925](./assets/image-20231220103800925.png) 

### Advanced Loss

$$
L(u, \alpha, c) = \frac{\vert 2-\alpha \vert}{\alpha} \times \left[
\left(
\dfrac{(x/c)^2}{\vert 2-\alpha \vert} + 1
\right)^{\alpha/2}
-1
\right]
$$

![image-20231228210108723](./assets/image-20231228210108723.png)

### Adaptive Loss

No hyper-parameter tuning!, as $\alpha$ is optimized for its most optimal value as well

If the selection of $\alpha$ wants to discount the loss for outliers, it needs to increase the loss for inliers

$$
\begin{aligned}
\text{J}'
&= -\log P(u \vert \alpha) \\
&= J(u, \alpha) + \log Z(\alpha)
\end{aligned}
$$

![image-20231228225321275](./assets/image-20231228225321275.png)

## Classification Loss

| Metric                                                       |                           Formula                            |     Range     | Preferred Value | Meaning                                                      |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :-----------: | :-------------: | ------------------------------------------------------------ |
| **Cross<br />Entropy**/<br />Log Loss/<br />Negative Log Likelihood | $-\sum\limits_c^C p_c \cdot \ln q_c$<br />such that $\sum p_i = \sum q_i$<br />where $p_i$ = Truth, $q_i$ = prediction, $C$ = number of classes | $[0, \infty]$ |  $\downarrow$   | Minimizing gives us $p=q$ for $n>>0$ (Proven using Lagrange Multiplier Problem)<br />Information Gain $\propto \dfrac{1}{\text{Entropy}}$<br />Entropy: How much information gain we have |

## Costs Functions $J(\theta)$

Aggregated penalty for entire set (mean, median) which is calculated once for each epoch, which includes loss function and/or regularization

This is the objective function on for our model to minimize
$$
J(\theta) = f( \ L(\theta) \ )
$$
where $f=$ summary statistic such as mean, etc

For eg:

- Mean(SE) = MSE, ie Mean Squared Error
- $\text{RMSE} = \sqrt{\text{MSE}}$
- Normalized RMSE = $\dfrac{\text{RMSE}}{y}$

### Robustness to outliers

In all the error metrics, we can replace mean with another summary statistic

- Median
- IQR
- Trimmed Mean

### Penalize number of predictors

Correction factor

$$
J'(\theta) = \dfrac{n}{n-k} \times J(\theta)
$$

###  Weighted Loss

$$
J'(\theta) = J(w_i \theta)
$$

where $J(\theta)$ is usual loss function

| Goal: Incorporate       | $w_i$                                                        | Prefer                                                       | Comment                                                      |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Asymmetry of importance | $\Big( \text{sgn}(u_i) - \alpha \Big)^2 \\ \alpha \in [-1, 1]$ | $\begin{cases} \text{Under-estimation}, & \alpha < 0 \\ \text{Over-estimation}, & \alpha > 0 \end{cases}$ |                                                              |
| Observation Error       | $\dfrac{1}{1 + \sigma^2_{y_i}}$<br />Where $\sigma^2_{y_i}$ is the uncertainty associated with each observation. | Observations with low uncertainty                            | Maximum likelihood estimation                                |
| Measurement Error       | $\dfrac{1}{1 + \sigma^2_X}$<br />Where $\sigma^2_X$ is the error covariance matrix | Observations with high input measurement accuracy            |                                                              |
| Heteroscedasticity      | $\dfrac{1}{1 + \sigma^2_{yi}}$<br />Where $\sigma^2_i$ is the uncertainty associated with each observation. | Observations in regions of low variance                      |                                                              |
| Observation Importance  | Importance                                                   | Observations with high domain-knowledge importance           | For time series data, you can use $w_i = \text{Radial basis function(t)}$<br/>- $\mu = t_\text{max}$<br/>- $\sigma^2 = (t_\text{max} - t_\text{min})/2$ |