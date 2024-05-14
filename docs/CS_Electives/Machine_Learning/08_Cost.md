# Cost Function

## Error

Using [MLE](06_MLE.md), we define error as

$$
u_i = \hat y_i - y_i \\
u_i' = \dfrac{u_i}{\sigma_{yi}}
$$

Usually we expect that $\sigma_{yi}=1$ and without any measurement noise

Bayes’ Error is the error incurred by an ideal model, which is one that makes predictions from true distribution $P(x,y)$; even such a model incurs some error due to noise/overlap in the distributions

### Total Regression

Total Least Squares

Types

- Deming Regression
- Orthogonal regression
- Geometric mean
- Method of moments
- Full total regression

Useful for when data has noise due to

- Measurement error
- Need for privacy etc, such as when conducting a salary survey.

$$
\begin{aligned}
L &= \left( \dfrac{u_i}{\sigma_{yi}} \right)^2  + \lambda \left( \dfrac{\hat x_i - x_i}{\sigma_{xi}} \right)^2 \\
\lambda &= \dfrac{\sigma^2(\text{known measurement error}_x)}{\sigma^2(\text{known measurement error}_y)} \\
\hat y_i &= \hat \beta_0 + \hat \beta_1 x_i \\
\hat x_i &= \dfrac{y_i - \beta_0}{\beta_1} \\
\text{or} \\
\hat x_i &= x_i + \dfrac{y_i - \hat y_i}{\partial f/\partial x_i} \dfrac{(\partial f/\partial x_i)^2 \sigma^2_{x_i}}{\sqrt{1+\sigma^2_{y_i, \text{eff}}}} \\
\sigma^2_{y_i, \text{eff}} &= \sigma^2_{y_i} + \left( \dfrac{\partial f}{\partial x_i} \right)^2
\end{aligned}
$$

How to get $\partial f/\partial x_i$?

1. Run regression ignoring $\sigma_{xi}$
2. Use this model fit to calculate $\partial f/\partial x_i \quad \forall i$ 
3. Calculate the effective variance $\sigma^2_{yi, \text{eff}} \quad \forall i$
4. Run weighted regression using 1/effective variance to weight $y_i$
5. Repeated steps until parameters converge (usually takes 1-2 iterations)

| Measurement Error of Regressor | $\lambda$ |            |
| ------------------------------ | --------- | ---------- |
| 0                              | 0         | OLS        |
| Same as Response               | 1         | Orthogonal |

![image-20231218180609992](./assets/image-20231218180609992.png)

![image-20231218180618882](./assets/image-20231218180618882.png)

#### Orthogonal Regression

- $\sigma_x = \sigma_y$
- Applied when you measure the same quantity with 2 different methods: for tool matching, calibration curves
- For straight-line model, $\sigma^2_{y_i, \text{eff}} \propto (1 + {\beta_1}^2)$

$$
\begin{aligned}
\sigma^2_{y_i, \text{eff}}
&= \sigma^2_y \left[
1 + \left(\dfrac{\partial f}{\partial x_i}\right)^2 \\
\right]
\\
\chi^2
&= \sum_{i=1}^n \left[ \dfrac{u_i}{\sqrt{1+{\beta_1}^2}} \right]^2
\end{aligned}
$$

#### Geometric Mean Regression

- Estimate slope as geometric mean of OLS slopes from $y \vert x$  and $x \vert y$
- $\beta_1 = \dfrac{s_y}{s_y}$

#### Method of Moments

If we know measurement error in $x:$ $\sigma_{\delta_x}$

Only good when $n>50$

$$
\begin{aligned}
\beta_1' &= \dfrac{\beta_1}{1 - \left(\dfrac{s_{\delta_x}}{s_x} \right)^2}
\\
\text{SE}(\beta_1') &= \dfrac{\beta_1}{\sqrt{n}} \sqrt{\dfrac{(s_x s_y)^2 + 2 (\beta_1 s^2_{\delta_x})^2}{(s_{xy})^2} - 1}
\end{aligned}
$$

#### Deming Regression

OLS of $y_i \vert \hat x_i$ produces same fit as Deming of $y_i \vert x_i$

Assumes that there is no model error: all uncertainty in $x$ and $y$ is due to measurement

If Deming Regression and Method of Moments give different estimates, then the model specification may be incorrect

## MSE vs Chi-Squared

- MSE($u_i$) may/may not $= \chi^2_\text{red}$
- MSE($u_i'$) $= \chi^2_\text{red}$

## Loss Functions ${\mathcal L}(\theta)$

$$
\text{Loss}_i = {\mathcal L}(\theta, u_i)
$$

- Penalty for a single point (absolute value, squared, etc)
- Should always be tailor-made for each problem, unless impossible
  - Need not be symmetric
  - Regression: Under-prediction and over-prediction can be penalized differently
  - Classification: False negative and false-positive can be penalized differently


### Properties of Loss Function

|                         |                                                              |                                               |
| ----------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| Non-negativity          | ${\mathcal L}(u_i) \ge 0, \quad \forall i$                              |                                               |
| No penalty for no error | ${\mathcal L}(0)=0$                                                     |                                               |
| Monoticity              | $\vert u_i \vert > \vert u_j \vert \implies {\mathcal L}(u_i) > {\mathcal L}(u_j)$ |                                               |
| Differentiable          | Continuous derivative                                        |                                               |
| Symmetry                | ${\mathcal L}(-u_i)={\mathcal L}(+u_i)$                                            | Not always required for custom loss functions |

###  Weighted Loss

Related to weighted regression
$$
{\mathcal L}'(\theta) = {\mathcal L}(w_i \theta)
$$

$$
\begin{aligned}
u'_i &= u_i \times \sqrt[a]{w_i} \\
\text{SE}(u')
&= s_{u'} \\
&= \sqrt{\dfrac{\sum_{i=1}^n w_i (u_i)^2}{n-k}} \\
&= \sqrt{\dfrac{\sum_{i=1}^n (u_i')^2}{n-k}}
\end{aligned}
$$

where

- ${\mathcal J}(\theta)$ = usual loss function
- $a=$ exponent of the loss function (square, etc)

| Goal: Address                                                | Action                                                       | Prefer                                                       | Comment                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Asymmetry of importance                                      | $w_i = \Big( \text{sgn}(u_i) - \alpha \Big)^2 \\ \alpha \in [-1, 1]$ | $\begin{cases} \text{Under-estimation}, & \alpha < 0 \\ \text{Over-estimation}, & \alpha > 0 \end{cases}$ |                                                              |
| Observation Error<br />Measurement/Process<br /><br />Heteroskedasticity | $\sigma^2_{yi}$<br />where $\sigma^2_{y_i}$ is the uncertainty associated with each observation | Observations with low uncertainty                            | Maximum likelihood estimation                                |
| Input Error<br />Measurement/Process                         | $w_i = \dfrac{1}{\sigma^2_{X}}$<br />where $\sigma^2_X$ is the uncertainty associated | Observations with high input measurement accuracy            |                                                              |
| Observation Importance                                       | $w_i=$ Importance                                            | Observations with high domain-knowledge importance           | For time series data, you can use $w_i = \text{Radial basis function(t)}$<br/>- $\mu = t_\text{max}$<br/>- $\sigma^2 = (t_\text{max} - t_\text{min})/2$ |

## Regression Loss

![image-20231218182207191](./assets/image-20231218182207191.png)

| Metric                                                     |                     ${\mathcal L}(u_i)$                      | Optimizing gives __ of conditional distribution |                       Preferred Value                        |     Unit      |        Range        | Signifies                                                    | Advantages<br />✅                                            | Disadvantages<br />❌                                         | Comment                                                      | $\alpha$ of advanced family | Breakdown Point | Efficiency |
| :--------------------------------------------------------- | :----------------------------------------------------------: | ----------------------------------------------- | :----------------------------------------------------------: | :-----------: | :-----------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- | --------------------------- | --------------- | ---------- |
| Indicator/<br />Zero-One/<br />Misclassification           | $\begin{cases} 0, & u_i = 0 \\ 1, & \text{o.w} \end{cases}$  | Mode                                            |                                                              |               |                     |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| BE<br />(Bias Error)                                       |                            $u_i$                             |                                                 | $\begin{cases} 0, & \text{Unbiased} \\ >0, & \text{Over-prediction} \\ <0, & \text{Under-pred} \end{cases}$ |  Unit of $y$  | $(-\infty, \infty)$ | Direction of error bias<br />Tendency to overestimate/underestimate |                                                              | Cannot evaluate accuracy, as equal and opposite errors will cancel each other, which may lead to non-optimal model having error=0 |                                                              |                             |                 |            |
| L1/<br />AE<br />(Absolute Error)/<br />Manhattan distance |                     $\vert  u_i  \vert$                      | Median                                          |                         $\downarrow$                         |  Unit of $y$  |    $[0, \infty)$    |                                                              | Robust to outliers                                           | Not differentiable at origin, which causes problems for some optimization algo<br />There can be multiple optimal fits<br />Does not penalize large deviations | MLE for $\chi^2$ for Laplacian dist                          |                             |                 | $74 \%$??? |
| L2/<br />SE (Squared Error)/<br />Euclidean distance       |                          ${u_i}^2$                           | Mean                                            |                         $\downarrow$                         | Unit of $y^2$ |    $[0, \infty)$    | Variance of errors with mean as MDE<br />Maximum log likelihood | Penalizes large deviations                                   | Sensitive to outliers                                        | MLE for $\chi^2$ for normal dist                             | $\approx 2$                 | $1/n$           | $100 \%$   |
| L3/<br />Smooth L1/<br />Pseudo-Huber/<br />Charbonnier    | $\begin{cases} \dfrac{u_i^2}{2 \epsilon}, & \vert u_i \vert < \epsilon \\ \vert u_i \vert - \dfrac{\epsilon}{2}, & \text{o.w} \end{cases}$ |                                                 |                         $\downarrow$                         |               |    $[0, \infty)$    |                                                              | Balance b/w L1 & L2<br />Prevents exploding gradients<br />Robust to outliers |                                                              | Piece-wise combination of L1&L2                              | 1                           |                 |            |
| Huber                                                      | $\begin{cases} \dfrac{u_i^2}{2}, & \vert u_i \vert < \epsilon \\ \lambda \vert u_i \vert - \dfrac{\lambda^2}{2}, & \text{o.w} \end{cases}$<br /><br />$(\lambda_\text{rec} = 1.345 \times \text{MAD}_u)$ |                                                 |                         $\downarrow$                         |               |    $[0, \infty)$    |                                                              | Same as Smooth L1                                            | Computationally-expensive for large datasets<br />$\epsilon$ needs to be optimized<br />Only first-derivative is defined | Piece-wise combination of L1&L2<br />$H = \epsilon \times {L_1}_\text{smooth}$<br />Solution behaves like a trimmed mean: (conditional) mean of two (conditional) quantiles |                             |                 | $95 \%$    |
| Log Cosh                                                   | $\Big\{ \log \big( \ \cosh (u_i) \ \big) \Big \} \\ \approx \begin{cases} \dfrac{{u_i}^2}{2}, & \vert u_i \vert \to 0 \\ \vert u_i \vert - \log 2, & \vert u_i \vert \to \infty \end{cases}$ |                                                 |                         $\downarrow$                         |               |                     |                                                              | Same as Smooth L1<br />Doesn’t require hyperparameter tuning<br />Double differentiable |                                                              |                                                              |                             |                 |            |
| Cauchy/<br />Lorentzian                                    | $\dfrac{\epsilon^2}{2} \times \log \Big[ 1 + \left( \dfrac{{u_i}}{\epsilon} \right)^2 \Big]$ |                                                 |                         $\downarrow$                         |               |                     |                                                              | Same as Smooth L1                                            | Not convex                                                   |                                                              | 0                           |                 |            |
| Log-Barrier                                                | $\begin{cases} - \epsilon^2 \times \log \Big(1- \left(\dfrac{u_i}{\epsilon} \right)^2 \Big) , & \vert u_i \vert \le \epsilon \\ \infty, & \text{o.w} \end{cases}$ |                                                 |                                                              |               |                     |                                                              |                                                              | Solution not guaranteed                                      | Regression loss $< \epsilon$, and classification loss further |                             |                 |            |
| $\epsilon$-insensitive                                     | $\begin{cases} 0, & \vert u_i \vert \le \epsilon \\ \vert u_i \vert - \epsilon, & \text{otherwise} \end{cases}$ |                                                 |                                                              |               |                     |                                                              |                                                              | Non-differentiable                                           |                                                              |                             |                 |            |
| Bisquare/<br />Welsch/<br />Leclerc                        | $\begin{cases} \dfrac{\lambda^2}{6 \left(1- \left[1-\left( \dfrac{u_i}{\lambda} \right)^2 \right]^3 \right)}, & \vert u_i \vert < \lambda \\ \dfrac{\lambda^2}{6}, & \text{o.w} \end{cases}$<br />$\lambda_\text{rec} = 4.685 \times \text{MAD}_u$ |                                                 |                         $\downarrow$                         |               |                     |                                                              | Robust to outliers                                           | Suffers from local minima (Use huber loss output as initial guess) | Ignores values after a certain threshold                     | $\infty$                    |                 |            |
| Geman-Mclure                                               |                                                              |                                                 |                                                              |               |                     |                                                              |                                                              |                                                              |                                                              | -2                          |                 |            |
| Quantile/Pinball                                           | $\begin{cases} q \vert u_i \vert , & \hat y_i < y_i \\ (1-q) \vert u_i \vert, & \text{o.w} \end{cases}$<br />$q = \text{Quantile}$ | Quantile                                        |                         $\downarrow$                         |  Unit of $y$  |                     |                                                              | Robust to outliers                                           | Computationally-expensive                                    |                                                              |                             |                 |            |
| Expectile                                                  | $\begin{cases} e (u_i)^2 , & \hat y_i < y_i \\ (1-e) (u_i)^2, & \text{o.w} \end{cases}$<br />$e = \text{Expectile}$ | Expectable                                      |                         $\downarrow$                         | Unit of $y^2$ |                     |                                                              |                                                              |                                                              | Expectiles are a generalization of the expectation in the same way as quantiles are a generalization of the median |                             |                 |            |
| APE<br />(Absolute Percentage Error)                       |       $\left \lvert  \dfrac{ u_i }{y_i}  \right \vert$       |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    |                                                              | Easy to understand<br />Robust to outliers                   | Explodes when $y_i \approx 0$<br />Division by 0 error when $y_i=0$<br />Asymmetric: $\text{APE}(\hat y, y) \ne \text{APE}(y, \hat y) \implies$ Penalizes over-prediction more than under-prediction<br />Sensitive to measurement units |                                                              |                             |                 |            |
| SMAPE<br />Symmetric APE                                   | $\left \lvert  \dfrac{ u_i }{\hat y_i + y_i}  \right \vert$  |                                                 |                                                              |               |                     |                                                              |                                                              |                                                              | Denominator is meant to be mean($\hat y, y$), but the 2 is cancelled for appropriate range |                             |                 |            |
| SSE<br />(Squared Scaled Error)                            |   $\dfrac{ 1 }{\text{SE}(y_\text{naive}, y)} \cdot u_i^2$    |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| ASE<br />(Absolute Scaled Error)                           | $\dfrac{ 1 }{\text{AE}(y_\text{naive}, y)} \cdot \vert u_i \vert$ |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| ASE Modified                                               | $\left \lvert \dfrac{ u_i }{y_\text{naive} - y_i} \right \vert$ |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    |                                                              |                                                              | Explodes when $\bar y - y_i \approx 0$<br />Division by 0 error when $\bar y - y_i \approx 0$ |                                                              |                             |                 |            |
| WMAPE<br />(Weighted Mean Absolute Percentage Error)       | $\dfrac{1}{n} \left(\dfrac{ \sum \vert  u_i \vert  }{\sum \vert  y_i  \vert}\right)$ |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    |                                                              | Avoids the  limitations of MAPE                              | Not as easy to interpret                                     |                                                              |                             |                 |            |
| MALE                                                       | $\vert \ \log_{1p} \vert \hat y_i \vert - \log_{1p} \vert y_i \vert \ \vert$ |                                                 |                                                              |               |                     |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| MSLE<br />(Log Error)                                      | $(\log_{1p} \vert \hat y_i \vert - \log_{1p} \vert y_i \vert)^2$ |                                                 |                         $\downarrow$                         | Unit of $y^2$ |    $[0, \infty)$    | Equivalent of log-transformation before fitting              | - Robust to outliers<br />- Robust to skewness in response distribution<br />- Linearizes relationship | - Penalizes under-prediction more than over-prediction<br />- Penalizes large errors very little, even lesser than  MAE (still larger than small errors, but weight penalty inc very little with error)<br />- Less interpretability |                                                              |                             |                 |            |
| RAE<br />(Relative Absolute Error)                         | $\dfrac{\sum \vert  u_i \vert}{\sum \vert y_\text{naive} - y_i  \vert}$ |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    | Scaled MAE                                                   |                                                              |                                                              |                                                              |                             |                 |            |
| RSE<br />(Relative Square Error)                           |   $\dfrac{\sum  (u_i)^2 }{\sum (y_\text{naive} - y_i)^2 }$   |                                                 |                         $\downarrow$                         |       %       |    $[0, \infty)$    | Scaled MSE                                                   |                                                              |                                                              |                                                              |                             |                 |            |
| Peer Loss                                                  | ${\mathcal L}(\hat y_i \vert x_i, y_i) - L \Big(y_{\text{rand}_j} \vert x_i, y_{\text{rand}_j} \Big)$<br />${\mathcal L}(\hat y_i \vert x_i, y_i) - L \Big(\hat y_j \vert x_k, y_j \Big)$<br />Compare loss of actual prediction with predicting a random value |                                                 |                                                              |               |                     | Actual information gain                                      | Penalize overfitting to noise                                |                                                              |                                                              |                             |                 |            |
| Winkler score $W_{p, t}$                                   |    $\dfrac{Q_{\alpha/2, t} + Q_{1-\alpha/2, t}}{\alpha}$     |                                                 |                         $\downarrow$                         |               |                     |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| CRPS<br />(Continuous Ranked Probability Scores)           |               $\overline Q_{p, t}, \forall p$                |                                                 |                         $\downarrow$                         |               |                     |                                                              |                                                              |                                                              |                                                              |                             |                 |            |
| CRPS_SS<br />(Skill Scores)                                | $\dfrac{\text{CRPS}_\text{Naive} - \text{CRPS}_\text{Method}}{\text{CRPS}_\text{Naive}}$ |                                                 |                         $\downarrow$                         |               |                     |                                                              |                                                              |                                                              |                                                              |                             |                 |            |

Robust estimators are only robust to non-influential outliers

### Outlier Sensitivity

![image-20231220103800925](./assets/image-20231220103800925.png) 

### Advanced Loss

$$
{\mathcal L}(u, \alpha, c) = \frac{\vert \alpha - 2 \vert}{\alpha} \times \left[
\left(
\dfrac{(u/c)^2}{\vert \alpha - 2 \vert} + 1
\right)^{\alpha/2}
-1
\right]
$$

If you don’t want to optimize for $c$, default $c=1$

![image-20231228210108723](./assets/image-20231228210108723.png)

![image-20240721225313029](./assets/image-20240721225313029.png)

- Monotonic wrt $\vert u \vert$ and $\alpha$: useful for graduated non-convexity
- Smooth wrt $u$ and $\alpha$
- Bounded first and second derivatives: no exploding gradients, easy preconditioning

### Adaptive Loss

No hyper-parameter tuning!, as $\alpha$ is optimized for its most optimal value as well

If the selection of $\alpha$ wants to discount the loss for outliers, it needs to increase the loss for inliers

$$
\begin{aligned}
\text{L}'
&= -\log P(u \vert \alpha) \\
&= {\mathcal L}(u, \alpha) + \log Z(\alpha)
\end{aligned}
$$

![image-20231228225321275](./assets/image-20231228225321275.png)

## Classification Loss

Should be tuned to control which type of error we want to minimize

- overall error rate
- false positive rate (FPR)
- false negative rate (FNR)

Imbalanced dataset: Re-weight loss function to ensure equal weightage for each target class

- Sample weight matching the probability of each class in the population data-generating distribution
  - For eg $\sum_{i} w_{ic} = \text{same}, \forall c$
    - $w_i = 1-f_c = 1-\dfrac{n_c}{n}$, where $n_i=$ no of observations of class $c$
  
- Modify loss function
- Under-sampling

| Metric                                                       | Formula                                                      |     Range     | Preferred Value | Meaning                                                      | Advantages                                                | Disadvantages                                                |
| ------------------------------------------------------------ | :----------------------------------------------------------- | :-----------: | :-------------: | ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------ |
| Indicator/<br />Zero-One/<br />Misclassification             | $\begin{cases} 0, & \hat y = y \\ 1, & \text{o.w} \end{cases}$ |   $[0, 1]$    |       $0$       | Produces a Bayes classifier that maximizes the posterior probability | Easy to interpret                                         | Treats all error types equally<br />Minimizing is np-hard<br />Not differentiable |
| Modified Indicator                                           | $\begin{cases} 0, & \hat y = y \\ a, & \text{FP} \\ b, & \text{FN} \end{cases}$ |               |       $0$       |                                                              | Control on type of error to min                           | Harder to interpret                                          |
| **Cross<br />Entropy**/<br />Log Loss/<br />Negative Log Likelihood/<br />Softmax | $-\sum\limits_c^C p_c \cdot \ln \hat p_c$<br />such that $\sum p_i = \sum \hat p_i$<br /><br />$-\ln \left( \dfrac{\exp(\hat p_i)}{\sum_{j=c}^C \exp(\hat p_c)} \right)$, where $i=$ correct class<br />$- \hat p_i + \ln \sum_{c=1}^C \exp(\hat p_c)$ | $[0, \infty]$ |  $\downarrow$   | Minimizing gives us $p=q$ for $n>>0$ (Proven using Lagrange Multiplier Problem)<br />Information Gain $\propto \dfrac{1}{\text{Entropy}}$<br />Entropy: How much information gain we have |                                                           |                                                              |
| Binary cross entropy/<br />Logistic                          | $-\log \Big(\sigma(-\hat y \cdot y_i) \Big) = \log(1 + e^{-\hat y \cdot y_i})$<br />$y, \hat y \in \{-1, 1 \}$ |               |                 |                                                              |                                                           |                                                              |
| Gini Index                                                   | $\sum\limits_c^C p_c (1 - \hat p_c)$                         |               |                 |                                                              |                                                           |                                                              |
| Hinge                                                        | $\max \{ 0, 1 - y_i \hat y_i \}$<br />$y \in \{ -1, 1 \}$    |               |                 | Equivalent to $L_1$ loss but only for predicting wrong class<br />Maximize margin | - Insensitive to outliers: Penalizes errors “that matter” | - Loss is non-differentiable at point<br/>- Does not have probabilistic interpretation |
| Exponential                                                  | $\exp (-\hat y \cdot y)$<br />$y \in \{ -1, 1 \}$            |               |                 | Basic $e^{\text{CE}}$                                        |                                                           | Sensitive to outliers                                        |
| KL (Kullback-Leibler) Divergence/<br />Relative entropy/<br />Cross Entropy - Entropy | $H(p, q) - H(p)$                                             |               |                 |                                                              |                                                           |                                                              |

![classification_losses](./assets/classification_losses.png)

Example for $y = 1$

![image-20240710131325987](./assets/image-20240710131325987.png)

## Clustering Loss

$$
{\mathcal L}(\theta) = D\Big(
x_i,
\text{centroid}(\hat y_i)
\Big)
$$

### Proximity Measures

- Similarity
- Dissimilarity
  - Distance measure (subclass)

#### Range

May be

- $[0, 1], [0, 10], [0, 100]$
- $[0, \infty)$

### Types of Proximity Measures

#### Similarity

For document, sparse data

- Jacard Similarity
- Cosine Similarity

#### Dissimilarity

For continuous data

- Correlation
- Euclidean

### IDK

| Attribute Type | Dissimilarity                                                | Similarity                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Nominal        | $\begin{cases} 0, & p=q \\1, &p \ne q \end{cases}$           | $\begin{cases} 1, & p=q \\ 0, &p \ne q \end{cases}$          |
| Ordinal        | $\dfrac{\vert  p-q \vert}{n-1}$<br />Values mapped to integers: $[0, n-1]$, where $n$ is the no of values | $1- \dfrac{\vert  p-q  \vert}{n-1}$                          |
| Interval/Ratio | $\vert p-q \vert$                                            | $-d$ <br /> $\dfrac{1}{1+d}$ <br /> $1 - \dfrac{d-d_\text{min}}{d_\text{max}-d_\text{min}}$ |

### Dissimilarity Matrix

**Symmetric** $n \times n$ matrix, which stores a collection of dissimilarities for all pairs of $n$ objects

- $d(2, 1) = d(1, 2)$

It gives the distance from every object to every other object

Something

Example

| Object<br />Identifier | Test 1 | Tets 2 | Test 3 |
| ---------------------- | ------ | ------ | ------ |
|                        |        |        |        |
|                        |        |        |        |
|                        |        |        |        |

Compute for test 2

|       |  1   |  2   |  3   |  4   |
| :---: | :--: | :--: | :--: | :--: |
| **1** |      |      |      |      |
| **2** |      |      |      |      |
| **3** |      |      |      |      |
| **4** |      |      |      |      |

### Distance between data objects

#### Minkowski’s distance

Let

- $a, b$ be data objects
- $n$ be no of attributes
- $r$ be parameter

The distance between $x,y$ is

$$
d(a, b) =
\left(
\sum_{k=1}^n
\vert  a_k - b_k  \vert^r
\right)^{\frac{1}{r}}
$$

| $r$      | Type of Distance                                             |                     $d(x, y)$                     | Gives                  | Magnitude of Distance | Remarks                               |
| -------- | ------------------------------------------------------------ | :-----------------------------------------------: | ---------------------- | --------------------- | ------------------------------------- |
| 1        | City block<br />Manhattan<br />Taxicab<br />$L_1$ Norm       |      $\sum_{k=1}^n \vert  a_k - b_k  \vert$       | Distance along axes    | Maximum               |                                       |
| 2        | Euclidean<br />$L_2$ Norm                                    | $\sqrt{ \sum_{k=1}^n \vert  a_k - b_k  \vert^2 }$ | Perpendicular Distance | Shortest              | We need to standardize the data first |
| $\infty$ | Chebychev<br />Supremum<br />$L_{\max}$ norm<br />$L_\infty$ norm |         $\max (\vert  x_k - y_k  \vert )$         |                        | Medium                |                                       |
|          | Makowski                                                     |                                                   |                        |                       |                                       |

Also, we have squared euclidean distance, which is used sometimes

$$
d(x, y) =
\sum_{k=1}^n |a_k - b_k|^2
$$

### Properties of Distance Metrics

| Property              | Meaning                         |
| --------------------- | ------------------------------- |
| Non-negativity        | $d(a, b) = 0$                   |
| Symmetry              | $d(a, b) = d(b, a)$             |
| Triangular inequality | $d(a, c) \le d(a, b) + d(b, c)$ |

### Similarity between Binary Vector

$M_{00}$ shows how often do they come together; $p, q$ do not have 11 in the same attribute

#### Simple Matching Coefficient

$$
\text{SMC}(p, q) =
\frac{
M_{00} + M_{11} (\text{Total no of matches})
}{
\text{Number of attributes}
}
$$

#### Jaccard Coefficient

We ignore the similarities of $M_{00}$

$$
\text{JC}(p, q) =
\frac{M_{11}}{M_{11} + M_{01} + M_{10}}
$$

### Similarity between Document Vectors

#### Cosine Similarity

$$
\begin{aligned}
\cos(x, y) &= \frac{
xy
}{
\vert  x \vert  \ \ \vert  y  \vert 
}
\sum_{i=1}^n x_i y_i \\
&= x \cdot y \\
\vert  x  \vert &= \sqrt{\sum_{i=1}^n x_i^2}
\end{aligned} 
$$

| $\cos (x, y)$ | Interpretation              |
| ------------- | --------------------------- |
| 1             | Similarity                  |
| 0             | No similarity/Dissimilarity |
| -1            | Dissimilarity               |

#### Document Vector

Frequency of occurance of each term

$$
\cos(d_1, d_2) =
\frac{d_1 d_2}{
||d_1|| \ \ ||d_2||
}
\sum_{i=1}^n d_1 d_2
$$

#### Tanimatto Coefficient/Extended Jaccard Coefficient

$$
T(p, q) =
\frac{
pq
}{
||p||^2 + ||q||^2 - pq
}
$$

## Costs Functions ${\mathcal J}(\theta)$

Aggregated penalty for entire dataset (mean, median) which is calculated once for each epoch, which includes loss function and/or regularization

This is the objective function on for our model to minimize
$$
{\mathcal J}(\hat y, y) = f( \ {\mathcal L}(\hat y, y) \ )
$$
where $f=$ summary statistic such as mean, etc

You can optimize

- location: mean, median, etc
- scale: variance, IQR, etc
- Combination of both

For eg:

- Mean(SE) = MSE, ie Mean Squared Error
- $\text{RMSE} = \sqrt{\text{MSE}}$
- Normalized RMSE = $\dfrac{\text{RMSE}}{\bar y}$

### RMSE

RMSE is a good balance between MSE and MAE, as it is similar to

- MSE: penalizes large deviations
- MAE: is in the same unit as $y$

### Bessel’s Correction

Penalize number of predictors
$$
\begin{aligned}
\text{Cost}_\text{corrected} &= \text{Cost}_\text{corrected} \times \dfrac{n}{\text{DOF}} \\
\text{DOF} &= n-k-e
\end{aligned}
$$

- where
  - $n=$ no of samples
  - $k=$ no of parameters
  - $e=$ no of intermediate estimates (such as $\bar x$ for variance)

- Modify accordingly for squares/root metrics

### Robustness to Outliers

- Median: Very robust, but very low efficiency
- Trimmed Mean: Does not work well for small sample sizes
- IQR
