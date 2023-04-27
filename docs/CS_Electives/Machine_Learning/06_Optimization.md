## Training Process

```mermaid
flowchart LR
i[Initialize θ] -->
calc[Calculate ŷ] -->
compare[Compare ŷ with y] -->
error[Calculate error] -->
a{Acceptable?} -->
|Yes| stop([Stop])

a -->
|No| change[Change θ to reduce cost] -->
calc
```

## Popular Optimization Algorithms

|                             | Meaning                     |
| --------------------------- | --------------------------- |
| Gradient Descent            |                             |
| Stochastic Gradient Descent | Randomized Gradient Descent |
| Adam Optimizer              |                             |


## Gradient Descent

Similar to trial and error

1. Start with some $\theta$ vector
2. Keep changing $\theta_0, \theta_1, \dots, \theta_n$ using derivative of cost function, until minimum for $J(\theta)$ is obtained - **Simultaneously**

$$
\theta_{\text{new}} =
\theta_{\text{prev}} -
\eta \ 
{\nabla J}
$$

|                       | Meaning                         |
| --------------------- | ------------------------------- |
| $\theta_{\text{new}}$ |                                 |
| $\theta_{\text{old}}$ |                                 |
| $\eta$                | Learning Rate                   |
| $\nabla J$            | Gradient vector of $J (\theta)$ |

$$
\frac{
\partial J(\theta)
}{
\partial Q
} =
\frac{1}{m}
\sum (\hat y - y) x
$$

$$
\begin{aligned}
\nabla J(\theta)
& \approx 1 \\
\begin{bmatrix}
\frac{ \partial J(\theta) }{\partial \theta_1} \\
\frac{ \partial J(\theta) }{\partial \theta_2} \\
\textcolor{orange}{
  \frac{ \partial J(\theta) }{\partial \theta_0}
}
\end{bmatrix}
&\approx 1
\end{aligned}
$$

Using equation

$$
\nabla J(\theta) =
\begin{bmatrix}
\Big(\sigma(\theta^T x) - y \Big) x_1 \\
\Big(\sigma(\theta^T x) - y \Big) x_2 \\
\Big(\sigma(\theta^T x) - y \Big)
\end{bmatrix}
$$

$$
\sigma(\theta^T x) = \sigma(\theta_1 x_1 + \theta_2 x_2 + \theta_0)
$$

Constant is at the end

## Learning Rate $\eta$

$0 < \eta < 1$

- Large value may lead to underfitting/overfitting
- Small value will lead to more time taken

Can be

- constant
- time-based decay

## Feature Scaling

Helps to speed up gradient descent by making it easier for the algorithm to reach minimum faster

Get every feature to approx $-1 \le x_i \le 1$ range

Atleast try to get $-3 \le x_i \le 3$ or $-\frac13 \le x_i \le \frac13$

#### Mean Normalisation

Make features have appox zero mean

Not for $x_0 = 1$

$$
x'_i = z_i = \frac{ x_i - \mu_i }{s_i}
$$

where:

- $\mu_i =$ average of all values of feature i in training set
- $s_i =$​ SD or range (max - min)
  of values of feature i in training set

### Regularized

$$
\begin{aligned}
\theta_0 &:= \theta_0 - \alpha \frac1m \sum_{i=1}^m ( h(x^{(i)}) - y^{(i)} ) x_0^{(i)} \\
\theta_j &:= \theta_0 - \alpha \left[ \frac1m \sum_{i=1}^m ( h(x^{(i)}) - y^{(i)} ) x_0^{(i)} + \frac{\lambda}m \theta_j \right] \\
&:= \theta_j \left(1 - \alpha \frac {\lambda}m \right) - \alpha \frac1m \sum_{i=1}^m \left( h(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\\
(j &= 1, 2, \dots, n)
\end{aligned}
$$

## Gradient Descent vs Normal Equation

|                 |         Normal Equation         |           Gradient Descent           |
| :-------------: | :-----------------------------: | :----------------------------------: |
|    $\alpha$     |                -                |               Required               |
|   Iterations    |                -                |               Required               |
| Feature scaling |                -                |               Required               |
|   Performance   |  Slow if $n > 10^4$<br />$O(n^3)$  | Works even for large $n$ <br /> $O(kn^2)$|
| Compatibility | Doesn’t work for classification |       Works for all algorithms       |
| No of features | Doesn't work when $X^TX$ is non-invertible | Works for all algorithms |

- Normal Equation
  - No need to choose $\alpha$
  - No need to iterate
- Gradient Descent
  - Works well even when n is large

