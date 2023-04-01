## Regression

is the process of predicting continuous values.

$$
\hat y = \theta_0 + \theta_1 x_1
$$

- $\theta_0$ is the value of $y$ when $x_1=0$
- $\theta_1$ shows the change of $y$, when $x_1$ increases by 1 unit

### Variables

#### Uni-Variate

1 independent to 1 dependent variable

#### Multi-Variate

Multiple independent, but only dependent variable

### Degree of Regression

#### Linear Regression

Best-fit is a straight line

#### Non-Linear Regression

Best-fit is a curve

## IDK

Perpendicular distance between any point from regression line

$$
\frac{g(x)}{||w||}
$$

## Behind the Scenes of Regression

$$
\begin{aligned}
\hat y &= ax + b \\
a &= \frac{
n \Sigma xy - \Sigma x \Sigma y
}{
n \Sigma x^2 - (\Sigma x)^2
} \\
b &= \frac{1}{n} \Big( \Sigma y - a \Sigma x \Big)
\end{aligned}
$$

The pivot point of the best fit line are $(\bar x, \bar y)$; which are the averages of $x$ and $y$

## Multiple Linear Regression

$$
\begin{aligned}
\hat y &=
\theta_0 +
\theta_1 x_1 +
\dots +
\theta_n x_n \\&=\theta^T X \\
\theta^T &= [\theta_0, \theta_1, \theta_2, \dots, \theta_n] \\X &= \begin{bmatrix}
1 \\x_1 \\x_2 \\
\dots \\x_n
\end{bmatrix}
\end{aligned}
$$

### 2 Variate Linear Regression

For this course, this is the max they can ask

$$
y = b_0 + b_1 x_1 + b_2 x_2
$$

## Something I missed



## Linear Basis Function

$$
\theta_j(x) =
\text{exp}
\left\{
\frac{-(x- \mu_j)^2}{2s^2}
\right\}
$$

These are local: small change in $x$ only affect nearby basis function.

- $u_j$ is control location
- $s$ is control scale(width)
