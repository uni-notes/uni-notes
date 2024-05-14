# Support Vector Machine

Goal: obtain hyperplane farthest from all sample points

Larger margins $\implies$ fewer dichotomies $\implies$ smaller $d_\text{vc}$

Margin = Distance b/w boundary and edge point closest to it

Note: $y \in \{ -1, 1 \}$, not $\{ 0, 1 \}$

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

### Limitations

- Does not work for Non-separable problems

## Soft Margin/Hinge Loss

Non-separable

Quantify margin violation: $y_i \hat y_i \le 1$ not satisfied
$$
y_i \hat y_i \ge 1-\epsilon_i
\\
\epsilon_i \ge 0 \\
\implies \text{Total violation} = \sum_{i=1}^n \epsilon_i
$$

$$
\arg \min_\theta \vert \vert \theta \vert \vert^2 + C \sum_{i=1}^n \epsilon_i \\
\text{Subject to: } \epsilon_i = \max \{ 0, 1 - y_i \hat y_i \} \quad \forall i
$$

$$
\arg \min_\theta
\vert \vert \theta \vert \vert^2
+ C \sum_{i=1}^n \max \{ 0, 1 - y_i \hat y_i \}
$$

Since it doesn’t matter which term we multiply by $c>0$, this is equivalent to
$$
\arg \min_\theta
\underbrace{L(y, \hat y)}_{\mathclap{\text{Hinge Loss}}}
+
\underbrace{\dfrac{\lambda}{2}\vert \vert \theta \vert \vert^2}_{\mathclap{\text{Regularizer}}}
$$

Regularization optimizes for max margin

### Gradient

$$
\nabla J(\theta) =
\begin{cases}
-y \cdot x, & y \hat y > 1 \\
0, & \text{o.w}
\end{cases}
$$

## Types

- Primal: better for large $n$
- Dual: better for large $k$

## Types of SVs

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

Interesting observation

- Features $\phi(x)$ are never used
- Only dot product $\phi(x)^T \phi(x)$ is used

We can compute dot product between $O(k^2)$ features in $O(k)$ time

Implication

- Faster for high-dimensions
- Can be applied for any model class that use dot products
  - Supervised: SVM, Linear Regression, Logistic regression
  - Unsupervised: PCA, Density Estimation

### Advantages

- Can handle high-dimensionality with lower computational cost

### Disadvantage

- Still computationally-expensive: $O(n^2)$, as we need compute distance $\phi(x_i, x_j) \quad \forall i, j$
