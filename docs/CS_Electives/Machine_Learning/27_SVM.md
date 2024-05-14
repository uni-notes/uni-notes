# Support Vector Machine

Goal: obtain hyperplane farthest from all sample points

Larger margins $\implies$ fewer dichotomies $\implies$ smaller $d_\text{vc}$

Margin = Distance b/w boundary and edge point closest to it

## Hard Margin

Linearly-separable

Consider $x_s$ be the nearest data point to the plane $\theta^T X + b=0$

Constrain $\theta: \vert \theta^T x_s + b \vert = 1$, so that we get a unique plane
$$
\underset{\theta, \gamma}{{\arg\max}} \ \gamma
\\
\text{subject to } \min_{s} \vert \hat y_s \vert = 1
%% \text{subject to } y \dfrac{\hat y}{\vert \vert \theta \vert \vert} \ge \gamma
$$
Since $\theta$ is $\perp$ to the plane in the $x$ space, margin = distance between $x_i$ and the plane $\theta^T X + b=0$ is given by
$$
\begin{aligned}
\gamma
&= \vert \hat \theta^T(x_i - x) \vert
\\
&= \left\vert \dfrac{\theta}{\vert \vert \theta \vert \vert} (x_i - x) \right\vert
\\
&= \dfrac{1}{\vert \vert \theta \vert \vert} \vert \theta^T x_i - \theta^T x \vert
\\
&= \dfrac{1}{\vert \vert \theta \vert \vert} \vert (\theta^T x_i + b) - (\theta^T x + b) \vert \\
&= \dfrac{1}{\vert \vert \theta \vert \vert} \vert 1 - 0 \vert
\\
&= \dfrac{1}{\vert \vert \theta \vert \vert}
\end{aligned}
$$

$$
\vert \hat y_s \vert = y_s \hat y_s
$$

Optimization problem can be re-written as
$$
\begin{aligned}
\underset{\theta}{\arg \max} \frac{1}{\vert \vert \theta\vert \vert}
&
\\
\text{Subject to constraint: }
&(\hat y y) \ge 1 \quad \forall i
\\
=&
\begin{cases}
\hat y \ge 1, & y_i > 0 \\
\hat y \le -1, & y_i < 0 
\end{cases}
\end{aligned}
$$

Re-writing again
$$
\begin{aligned}
\underset{\theta}{\arg \min} {\vert\vert \theta \vert\vert}^2
&
\\
\text{Subject to constraint: }
&(\hat y y) \ge 1 \quad \forall i
\\
=&
\begin{cases}
\hat y \ge 1, & y_i > 0 \\
\hat y \le -1, & y_i < 0 
\end{cases}
\end{aligned}
$$

Re-writing again
$$
\begin{aligned}
&\underset{\theta}{\arg \min} \ {\vert\vert \theta \vert\vert}^2
- \sum_\mathclap{s \in \text{Sup Vec}}\alpha_s \Big( \hat y y - 1 \Big) \\
& \max \alpha_s \ge 0 \quad \forall s
\end{aligned}
$$

## Soft Margin

Non-linearly-separable

Quantify margin violation: $y_i \hat y_i \le 1$ not satisfied
$$
y_i \hat y_i \ge 1-\epsilon_i; \epsilon_i \ge 0 \\
\implies \text{Total violation} = \sum_{i=1}^n \epsilon_i
$$

$$
\arg \min_\theta \vert \vert \theta \vert \vert^2 + C \sum_{i=1}^n \epsilon_i \\
\text{Subject to: } \\
y_i \hat y_i \ge 1-\epsilon_i
\quad \forall i \\
\epsilon_i \ge 0
\quad \forall i
$$

Types of SVs

|              | Margin SV    | Non-Margin SV |
| ------------ | ------------ | ------------- |
| $\alpha_s$   | $\in (0, C)$ | $= C$         |
| $\epsilon_i$ | $=0$         | $>0$          |

## Generalization

Since the complexity of the plane only depends on the support vectors, $d_\text{vc} = \text{\# of SVs}$
$$
\mathbb{E} [E_\text{out}] \le \dfrac{\mathbb{E}  [\text{\# of SV}]}{n-1}
$$

## Kernel Trick

Complex $h$, but still simple $H$, as complexity of plane only depends on support vectors

Kernel function: $\phi(x, x')$ is valid $\iff$

- It is symmetric
- Mercer’s condition: Matrix $[k(x_i, x_j)]$ is +ve semi-definite

Linear transformation function for Non-Linearly-Separable

For eg, to increase the dimensionality, we can use $\phi(x) = (x, x^2)$

| Kernel Function                                              | $\phi(x)$                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Linear                                                       | $x$                                                          |
| Polynomial                                                   | $(mx+c)^n$                                                   |
| Gaussian                                                     | $\exp \left( \dfrac{-\vert  x-y  \vert^2}{2 \sigma^2} \right)$ <br /> where $\sigma^2 =$ Variance of sample |
| RBF<br />(Radial Basis Function)<br />Most powerful, but not necessary in most cases | $\exp( -\gamma \vert  x_i - x_j  \vert^2 )$                  |
