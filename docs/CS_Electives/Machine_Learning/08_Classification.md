# Classification

deals with categorical data

## Classification

Supervised learning technique, which works with labelled dataset

In a classication dataset, you have

- Independent attributes $<A_1, A_2, A_3>$

- Discrete Target attribute

- Tuples/records/data objects/sample instance

  Tuple $= (x, y)$

    - $x =$ Feature Vector
    - $y =$ Class Label

### Task

To build a model that maps a faeture set to the corresponding class label

### Algorithms

Decision Trees, Naive Bayes, Rule-based

### Types

|                      | Eager Learner                                        | Lazy Learner                                 |
| -------------------- | ---------------------------------------------------- | -------------------------------------------- |
| During<br />Training | Learns relationship between class label & attributes | Stores traing records                        |
| During<br />Testing  |                                                      | Perform computations to classify test record |
| Example              | - Decision Tree<br />- Rule-Based Classifier         | - Nearest-neighbor classifier                |

## Process

1. Training (Induction)
2. Validation
3. Testing (Deduction)

If a model performs well for training data as well as unseen testing data, the model has **generalization ability**

Validation Set helps understand if the hyperparameters of the model are appropriate, and check over-fitting and under-fitting

Train-Validation-Test set split is usually 80:10:10

## Errors

1. Training Error
2. Testing Error
3. Generalization

A good model has 

## Model Overfitting

## Confusion Matrix

$n \times n$ matrix, where $n$ is the number of classes

|      |                |
| ---- | -------------- |
| TP   | True Positive  |
| TN   | True Negative  |
| FP   | False Positive |
| FN   | False Negative |

Actual Total number of

- +ve samples = TP + FN
- -ve samples = FP + TN

## Classification Accuracy Metrics

To analyze the classification

## Binary Classification

2-class classification

![confusion_matrix_True_False_Positive_Negative](./../assets/confusion_matrix.png){ loading=lazy }

## Multi-Class Classification

### Confusion Matrix with respect to A

|      | A    | B    | C    |
| ---- | ---- | ---- | ---- |
| A    | TP   | FN   | FN   |
| B    | FP   | TN   | TN   |
| C    | FP   | TN   | TN   |

## Classification Report of `sklearn`

Same thing taught in ML

Weighted average is used for imbalanced dataset

## Missed this class

[4. CS F415 Data Mining Decision Trees.pptx](../assets/4. CS F415 Data Mining Decision Trees.pptx) 

## $k$-fold Cross Validation

Each re

- Train on 

## Step Function

$$
\hat y =
\begin{cases}
0, & \hat y < 0.5 \\
1, & \hat y \ge 0.5
\end{cases}
$$

### Logistic Regression

Using Sigmoid Function

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

## Cross-Entropy Error Function

Related to information theory

Entropy: How much information gain we have

Information Gain $\propto \frac{1}{\text{Entropy}}$

$$
\begin{aligned}
C(\hat y, y)
&=
\frac{1}{2}
(\hat y - y) ^2 \\
J(\theta)
&= \frac{1}{m}
\sum_{i=1}^m
\text{Cost}(\hat y, y)
\end{aligned}
$$

$$
\begin{aligned}
C(\hat y, y)
&= \begin{cases}
- \log \vert  1 - \hat y  \vert , & y=0 \\
- \log \vert  \hat y  \vert , & y=1
\end{cases} \\
J(\theta)
&= \frac{-1}{m}
\sum
\end{aligned}
$$

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

## Bayesian Classifier

Called as ‘Naive’ classifier, due to following assumptions

- Empirically-proven
- Scales very well

## Bayesian Rule

$$
P(C | X) = \frac{
P(X|C) \times P(C)
}{
P(X)
}
$$

Posterior depends on

- Likelihood
- Prior

$$
\text{Posterior} =
\frac{
something
}{
something
}
$$

### MAP Rule

**M**aximum **A** **P**osterior

Helps us decide the class during test phase

Assign $x$ to $c^*$ if $P(C=c^* | X=x) > P(C=c|X=x)$

## Naive Bayes Classification

Calculate posterior probability, based on assumption that all input attributes are conditionally-independent

### Drawbacks

1. Doesn’t work for continuous independent variable
   1. We need to use Gaussian Classifier
2. Violation of Independence Assumption
3. Zero outlook

## Gaussian Classifier

Used for classifying continuous data

$$
\begin{aligned}
P(C|x)
& \propto P(x|C) \times P(C) \\
& \propto N(x; \mu_c, \sigma^2_c) P(C) \\
\underbrace{P(C)}_{\text{Posterior}}
& \propto
\underbrace{
\frac{1}{\sqrt{2\pi \sigma^2_c}}
\ \exp \left(
\frac{-(x-\mu_c)^2}{2\sigma^2_c}
\right)
}_{\text{Likelihood}}
\underbrace{P(C)|x}_{\text{Prior}}
\end{aligned}
$$

However equation is not used as it is; we take $\log$ on both sides and find **log likehood**

$$
\begin{aligned}
\text{LL}(x|C)
&= \text{LL}(x|\mu_c, \sigma_c^2) \\
&= \ln P(x | \mu_c, \sigma_c^2) \\
&= \ln \left[
\frac{1}{\sqrt{2 \pi \sigma_c^2}}
\right]
 \ \exp somethign
\end{aligned}
$$

$$
\text{LL} \underbrace{(C|x)}_\text{Posterior} =
\text{LL} \underbrace{(x|C)}_\text{Likelihood} +
\text{LR} \underbrace{(C)}_\text{Posterior}
$$

### 2 Classes

$$
\begin{aligned}
\ln \frac{P(C_1 | x)}{P(C_2 | x)}
&= \ln P(C_1 | x) - \ln P(C_2 | x) \\
&= \frac{-1}{2} ()
\end{aligned}
$$

- If log ratio $\ge 0$, assign to $C_1$
- If log ratio $<0$, assign to $C_2$

We need to ensure that we have equal sample of both classes, so that the prior probabilities of both the classes in the formula is the same.

## Confusion Matrix

Diagonal elements are True Positive

- 0.97777778
- 0.97777778
- 0.98
- 0.98518519

- 1
- 1
- 1
- 1

## Classifcation Accuracy Measures

### Jacquard Index

$$
\begin{aligned}
J(y, \hat y)
&= \frac{|y \cap \hat y|}{|y \cup \hat y|} \\
&= \frac{|y \cap \hat y|}{|y| + |\hat y| - |y \cap \hat y|}
\end{aligned}
$$

### F1 Score

Same as Data Mining

[Classification Accuracy Metrics](../Data_Mining/07_Classification.md#Classification Accuracy Metrics)

### Classification Report

```python
classfication_report(y_test, predictions)
```

#### Macro Average

$$
\begin{aligned}
\text{Total Macro Average (Recall)}
&= \frac{\sum \text{Recall of each class}}{\text{No of classes}} \\
\text{Macro Average of each class (Recall)}
&= \text{Recall of that class}
\end{aligned}
$$

$$
\begin{aligned}
\text{Total Macro Average (Precision)}
&= \frac{\sum \text{Precision of each class}}{\text{No of classes}} \\
\text{Macro Average of each class (Precision)}
&= \text{Precision of that class}
\end{aligned}
$$

$$
\begin{aligned}
\text{Total Macro Average (F1 Score)}
&= \frac{\sum \text{F1 Score of each class}}{\text{No of classes}} \\
\text{Macro Average of each class (F1 Score)}
&= \text{F1 Score of that class}
\end{aligned}
$$

#### Weighted Average

$$
\begin{aligned}
&\text{Weighted Average (Recall)} \\
&= \frac{
\sum \Big( \text{Recall of each class} \times \text{Support of each class} \Big)
}{\text{Size of sample}}
\end{aligned}
$$

$$
\begin{aligned}
&\text{Weighted Average (Precision)} \\
&= \frac{
\sum \Big( \text{Precision of each class} \times \text{Support of each class} \Big)
}{\text{Size of sample}}
\end{aligned}
$$

$$
\begin{aligned}
&\text{Weighted Average (F1 Score)} \\
&= \frac{
\sum \Big( \text{F1 Score of each class} \times \text{Support of each class} \Big)
}{\text{Size of sample}}
\end{aligned}
$$

### Log Loss/Cross-Entropy Error Function

Performance of a classifier where the predicted output is a probability value between 0 and 1.

$$
\text{Log Loss} =
\frac{-1}{n} \sum \Big[
y \log(\hat y) + (1-y) \log (1 - \hat y)
\Big]
$$

## Case-Based Reasoning

==Different from re-inforcement learning==

```mermaid
flowchart LR
np[New Problem] -->
rc[Retrieved Cases]
```

## Support Vector Machine

Goal is to obtain hyperplane farthest from all sample points

$$
\begin{aligned}
\text{Distance } & \text{between edge point and line} \\
&= \frac{|w^t x_i + w_0|}{||w||} \\
&=\frac{1}{||w||} \\
\implies m &= \frac{2}{||w||}
\end{aligned}
$$

Goal is to maximize ‘margin’ $m$ (distance between classes), subject to the following constraints

$$
\begin{cases}
w^t x_i + w_0 \ge 1, & x_i > 0 \\
w^t x_i + w_0 \le -1, & x_i <0 
\end{cases}
$$

In other words, we need to minimize cost function

$$
J(\theta) = \frac{1}{2} ||w||^2
$$

We can derive through linear-programming

## For Linearly-Separable

1. Plot sample points

2. Find support vectors (points that are on border of other class)

3. Find augmented vectors with bias = 1

$$
s_1 = \begin{pmatrix} 0 \\
 1 \end{pmatrix}
\implies
\tilde{s_1} = \begin{pmatrix} 0 \\
 1 \end{pmatrix}
$$

4. Find values of $\alpha$, assuming that

   - $+ve = +1$
   - $-ve = -1$

$$
\begin{aligned}
\alpha_1 \tilde{s_1} \cdot \tilde{s_1} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_1} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_1}
&= -1 \\
\alpha_1 \tilde{s_1} \cdot \tilde{s_2} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_2} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_2}
&= 1 \\
\alpha_1 \tilde{s_1} \cdot \tilde{s_3} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_3} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_3}
&= 1
\end{aligned}
$$

5. Find $w_i$

$$
w_i =
$$

6. Something

## Kernel function $\phi(x)$ 

Linear transformation function for Non-Linearly-Separable

For eg, to increase the dimensionality, we can use $\phi(x) = (x, x^2)$

| Kernel Function                                              | $\phi(x)$                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Linear                                                       | $x$                                                          |
| Polynomial                                                   | $(kx+c)^n$                                                   |
| Gaussian                                                     | $\exp \left( \dfrac{-\vert  x-y  \vert^2}{2 \sigma^2} \right)$ <br /> where $\sigma^2 =$ Variance of sample |
| RBF<br />(Radial Basis Function)<br />Most powerful, but not necessary in most cases | $\exp( -\gamma \vert  x_i - x_j  \vert^2 )$                          |

## KNN Classifier

represents each record as a datapoint with $m$ dimensional space, where $m$ is the number of attributes

$k$ Nearest Neighbor

Class label is the majority label of $k$ nearest neighbors

### Training

#### Requirements

- Set of labelled records
- **Normalized** Proximity/Distance metric
    - Min-Max normalization
    - $Z$-normalization
- Value of $k$ (preferably $k$ is odd)

### Steps

- Compute distance between test record and all train records
- Identify $k$ neighbors of test records
  (Low distance=high similarity)
- Use majority voiting to find the class label of test sample

### Choosing Value of $k$

| $k$       | Problems                                        |
| --------- | ----------------------------------------------- |
| too small | Overfitting<br />Susceptible to noise           |
| too large | Underfitting<br />Susceptible to far-off points |

1. Use a test set
2. Let $k = 3$
3. Record error rate of classifier/accuracy
4. $k = k+2$
5. Repeat steps 3 and 4, until value of $k$ for which
   1. error is min
   2. accuracy is maximum

### Regression

KNN can be used for regression as well

Predicted value will be the average of the continuous labels of $k$-nearest neighbors

## Distance-Weighted KNN Classifier

Closer points are given larger weights for the majority voting scheme
