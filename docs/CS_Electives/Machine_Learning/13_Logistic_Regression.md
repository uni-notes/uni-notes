# Logistic Regression

Uses Sigmoid Function

$$
\begin{aligned}
\hat y
&= \sigma(\theta^T x) \\
&= \frac{1}{1 + e^{(\theta^T x)}} \\
&= P(y=1|x)
\end{aligned}
$$

#### Cases to use

1. For 2 classes
     - 0/1
     - Yes/No
     - +ve/-ve
2. Probability of prediction is required
3. Data is linearly-seperable
4. Understand impact of factor?

## Decision Boundary/Surface

The boundary/surface that separates different classes

Generated using decision function

If we have $d$ dimensional data, our decision boundary will have $(d-1)$ dimensions

## Linear Separability

Means the ability to separate points of different classes using a line, with/without a non-linear activation function

$$
f(u) = \begin{cases}
1, & u \ge 0 \\
0, & \text{otherwise}
\end{cases}
$$

| Logic Gate | Linearly-Separable? |
| :--------: | :-----------------: |
|    AND     |          ✅          |
|     OR     |          ✅          |
|    XOR     |          ❌          |
|    XOR     |          ❌          |

![Linear Separability of Logic Gates](../assets/linear_separability.svg)

## Discrimant Function

Functions which takes an input vector $x$ and assignts it to one of the $k$ classes

## 2-Class Classification

Consider $y(x) = w^T x + w_0$

$\perp$ Distance of $x$ in $w$ direction $= \frac{w^T x}{||w||}$

Something $= \frac{- w_0}{||w||}$

## Multi-Class Classification

|                   | One-vs-Rest                                            | One-vs-One                               |
| ----------------- | ------------------------------------------------------ | ---------------------------------------- |
| No of classifiers | $k-1$                                                  | $\frac{k(k-1)}{2}$                       |
| Limitation        | Some point may have multiple classes/no classes at all | Multiple classes assigned to some points |

### Logistic Regression

$$
y_k(x) = {w_k}^T x + {w_k}_0
$$

$$
y_k(x) =
(
\underset{\text{Classes}}{w_k}
-
\underset{\text{Features}}{w_j}
)^T x + ({w_k}_0 - {w_j}_0)
$$

Decision of such a discriminant function is always singly-connected and convex.

### LDA

Linear Discriminant Analysis, using Fisher Linear Discriminant

Maximizes separation using multiple classes, by seeking a projection that best **discriminates** the data

It is also used a pre-processing step for ML application

#### Goals

- Find directions along which the classes are best-separated (ie, increase discriminatory information)
    - Maximize inter-class distance
    - Minimize intra-class distance
- It takes into consideration the scatter(variance) **within-classes** and **between-classes**

#### Steps

1. Find within-class Scatter/Covariance matrix

    $S_w = S_1 + S_2$

    - $S_1 \to$ Covariance matrix for class 1
    - $S_2 \to$ Covariance matrix for class 2

$$
S_1 = \begin{bmatrix}
\text{cov}(x_1, x_1) & \text{cov}(x_1, x_2) \\
   \text{cov}(x_2, x_1) & \text{cov}(x_2, x_2)
\end{bmatrix}
$$

$$
\begin{aligned}
\text{Cov}(x_j, x_k) &= \frac{1}{n_j - 1} \sum_{i=1, x \in C_j}^{n_1} (x_i - \mu_1)(x_i - \mu_1) \\
\text{Cov}(x_1, x_1) &= \frac{1}{n_1 - 1} \sum_{i=1, x \in C_1}^{n_1} (x_i - \mu_1)^2
\end{aligned}
$$

2. Find between-class scatter matrix
   
$$
S_B =
(\mu_1 - \mu_2)
(\mu_1 - \mu_2)^T
$$

3. Find [Eigen Value](#Eigen-Value)

4. Find [Eigen Vector](#Eigen-Vector)

5. Generate LDA Projection [Normalized Eigen Vector](#Normalized-Eigen-Vector)

6. Generate LDA score (projected value) in reduced dimensions
   
$$
\text{LDA Score} = x_1 v_1 + x_2 v_2
$$

### Eigen Value

$$
| A - \lambda I | = 0 \\
|S_w^{-1} S_B - \lambda I| = 0
$$

- $\lambda =$ Eigen Value(s)
    - If we get multiple eigen values, we only take the highest eigen value
    - It helps preserve more information. How??
- $I =$ Identity Matrix

We are taking $A=S_w^{-1} S_B$ because taking $S_w^{-1}$ helps us maximize $\frac{1}{x}, x \in S_w$

- Hence $x$ is minimized
- Thereby, within-class distance is minimized

### Eigen Vector

$$
(S_w^{-1} S_B - \lambda I) 
\textcolor{hotpink}{V}
= 0
$$

- $\lambda =$ Highest eigen value
- $V =$ Eigen Vector

### Normalized Eigen Vector

$$
V_\text{norm} =
\begin{bmatrix}
\frac{v_1}{\sqrt{v_1^2 + v_2^2}} \\
\frac{v_2}{\sqrt{v_1^2 + v_2^2}}
\end{bmatrix}
$$

## Least Squares vs Logistic Regression

Least Squares method is sensitive to outliers, due to large deviation and high cost function

Hence, logistic regression using sigmoid function is better for classification

## Types of Models

### Discrimative Model

Depend on simple conditional probabilities

- Logistic Regression
- Decision Tree
- Random Forest

### Generative Model

- Bayesian Classifier
- Gaussian Classifier
