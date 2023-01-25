## Regression

used to predict for the dependent variable on the basis of past information available on dependent and independent variables.

The estimated regression line is given by

$$
\begin{align}
\hat y &= b_0 + b_1 x \\b_1 &= \frac{
	n \ \sum (xy) - \sum x \sum y
}{
	n \ \sum x^2 - \Big( \sum x \Big)^2
} \\b_0 &= \bar y - b_1 \bar x \\
\bar x &= \frac {\sum x} n \\
\bar y &= \frac {\sum y} n
\end{align}
$$

| Term     | Meaning              |
| -------- | -------------------- |
| $y$      | dependent variable   |
| $x$      | independent variable |
| $b_0$    | y-intercept          |
| $b_1$    | slope                |
| $\hat y$ | estimated value      |
| $\bar x$ | mean of $x$          |
| $\bar y$ | mean of $y$          |

## Correlation

gives the degree of linear relationship between the 2 variables $x$ and $y$
$-1 \le r \le +1$

$$
r = \frac{
	n \sum(xy) - \sum x \sum y
}{
	\sqrt{ n \sum (x^2) - \big(\sum x \big)^2 }
	\sqrt{ n \sum (y^2) - \big(\sum y \big)^2 }
}
$$

|   Type    |    Correlation    |
| :-------: | :---------------: |
|   Weak    |   $|r| \le 0.5$   |
| Moderate  | $0.5 < |r| < 0.8$ |
|  Strong   |   $|r| \ge 0.8$   |
|           |                   |
| Directly  |      $r > 0$      |
| Inversely |      $r < 0$      |

## Coefficient of Determination

$R^2$ value is used for non-linear regression. It shows how well data fits within the regression.

It has a range of $[0, 1]$. Higher the better.

$$
\begin{align}
R^2 &= 1 - \frac{ \text{SS}_{res} }{ \text{SS}_{tot} } \\
\text{SS}_\text{res} &= \sum\limits_{i=1}^n (y_i - \hat y)^2 \\
\text{SS}_\text{tot} &= \sum\limits_{i=1}^n (y_i - \bar y)^2 \\
\bar y &= \frac{1}{n} \sum\limits_{i=1}^n y_i
\end{align}
$$

where

| Symbol                 | Meaning                                                      |
| ---------------------- | ------------------------------------------------------------ |
| $\text{SS}_\text{res}$ | Residual sum of squares                                      |
| $\text{SS}_\text{tot}$ | Total sum of squares<br />Proportional to variance of the data |

