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
\begin{align}
\hat y &= ax + b \\
a &= \frac{
n \Sigma xy - \Sigma x \Sigma y
}{
n \Sigma x^2 - (\Sigma x)^2
} \\
b &= \frac{1}{n} \Big( \Sigma y - a \Sigma x \Big)
\end{align}
$$

The pivot point of the best fit line are $(\bar x, \bar y)$; which are the averages of $x$ and $y$

## SEE (Standard Error of Estimate)

This is used for interpretation of the regression

$$
\rm{SEE} =
\sqrt{\frac{
\sum(\hat y - y)^2
}{
n-2
}}
$$
|  SEE  | Satisfactory? | Remark                      |
| :---: | ------------- | --------------------------- |
| $<=1$ | ✅             |                             |
| $>1$  | ❌             | More training data required |

## Model Evaluation

We don’t test the model on the same we trained it with, because it will give high in-sample accuracy, but may give low out-of-sample accuracy(which is really what we want).

Out-of-sample accuracy is the accuracy of the model when tested when never-before-seen data.

### Train-Test Split

The training and test set should be mutually-exclusive, to ensure good out-of-sample accuracy. Usually split it as 80%-20%

Then, after evaluation you should train your model with the testing data afterwards.

However, this will not work well all the time, as this will be dependent ; especially for realtime data, where the model is sensitive to the data.

### $k$-Fold Cross Validation

$k$ is called as decision parameter.

In this course, we are doing 4-fold. This is the most common evaluation model.

- Split the dataset into 4 random groups
- Do the 80-20% split for each group
- Take average of all accuracies

## Error Evaluation/Cost Functions

Lower the error, the better

But higher the $R^2$, better the model

Helps measuring accuracy, which depends on amount of prediction errors

| Metric | Full Form                    |                           Formula                            | Preferred Value |
| :----: | ---------------------------- | :----------------------------------------------------------: | :-------------: |
|  MAE   | Mean Absolute Error          |         $\frac{1}{n} \sum_{i=1}^n | \hat y_i - y_i|$         |  $\downarrow$   |
|  MSE   | Mean Squared Error           |        $\frac{1}{n} \sum_{i=1}^n (\hat y_i - y_i)^2$         |  $\downarrow$   |
|  RMSE  | Root Mean Square Error       |     $\sqrt{\frac{1}{n} \sum_{i=1}^n (\hat y_i - y_i)^2}$     |  $\downarrow$   |
|  RAE   | Relative Absolute Error      | $\frac{\sum_{i=1}^n |y_i - \hat y|}{\sum_{i=1}^n |y_i - \bar y|}$ |  $\downarrow$   |
|  RSE   | Relative Square Error        | $\frac{\sum_{i=1}^n (y_i - \hat y)^2}{\sum_{i=1}^n (y_i - \bar y)^2}$ |  $\downarrow$   |
| $R^2$  | Coefficient of Determination |                       $1 - \text{RSE}$                       |   $\uparrow$    |

## Multiple Linear Regression

$$
\begin{align}
\hat y &=
\theta_0 +
\theta_1 x_1 +
\dots +
\theta_n x_n \\&=\theta^T X \\
\theta^T &= [\theta_0, \theta_1, \theta_2, \dots, \theta_n] \\X &= \begin{bmatrix}
1 \\x_1 \\x_2 \\\dots \\x_n
\end{bmatrix}
\end{align}
$$

### 2 Variate Linear Regression

For this course, this is the max they can ask

$$
y = b_0 + b_1 x_1 + b_2 x_2

$$

$$
\begin{align}
b_1 &= \\b_2 &= \\b_0 &= \hat y - b_1 \overline{x_1} - \overline{x_2} \\

\sum {x_1}^2 &=
\sum(x_1 x_1) - \frac{\sum x_1\sum x_1}{n} \\
\sum {x_1}^2 &= \\\sum {x_2}^2 &= \\\sum {x_1 y} &= \\\sum {x_2 y} &= \\\sum {x_1}{x_2} &= \\
\end{align}
$$

## Bias vs Variance

We want **low value** of both

|         |                         Bias                          |                     Variance                      |
| :-----: | :---------------------------------------------------: | :-----------------------------------------------: |
| Meaning | Predicted value - Actual value<br />(Basically error) |    Difference of predictions, with each other     |
| Formula |                   $E[f(x)] - f(x)$                    | $E \Bigg[ \ \Big(f(x) - E[f(x)] \ \Big)^2 \Bigg]$ |

$$
\text{MSE} = \text{Bias} + \text{Variance}
$$

## Fitting

|          |                        Under-Fitting                         |                         Over-Fitting                         |
| -------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Meaning  |                                                              |                                                              |
| Variance |                              ⬇️                               |                              ⬆️                               |
| Solution | Increase model complexity<br />Increase training data<br />Remove noise from data<br />Inc no of features | Cross-Validation<br />More training data<br />Feature Reduction<br />Early Stopping<br />Regularization |

## Regularization

Reduce errors by fitting the function appropriately on the given training set and avoid overfitting.

This is done by adding a penalty term in the error function.

Helps reduce the variance.

|                |                 $L_1$                  |                    $L_2$                    | $L_3$                                                        |
| -------------- | :------------------------------------: | :-----------------------------------------: | ------------------------------------------------------------ |
| Common Name    |                 Lasso                  |                    Rigde                    | Lasso-Ridge                                                  |
|                |    Eliminates feature(s) completely    | Reduce/Normalize the effect of each feature |                                                              |
|                |       Helps in feature selection       |         Scale down the coefficients         |                                                              |
| Error Function | $RSS + \lambda \sum_{j=1}^m |\beta_i|$ |   $RSS + \lambda \sum_{j=1}^m \beta_i ^2$   | $RSS + \lambda_1 \sum_{j=1}^m  + \lambda_2 \sum_{j=1}^m \beta_i ^2$ |

### Nom

$| |$ is called as ‘nom’?

$$
||w^2|| = \sum |w^2|

$$
## RSS

Residual Sum of Squares is a type of error.

$$
\text{RSS} =
\sum_{i=1}^n
\left(
y_i - 
\left(
\beta_0 + \sum_{j=1}^m \beta_j x_{ij}
\right)
\right)^2

$$

## Hyperparameters

Parameters that affect the prediction of a model.

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
