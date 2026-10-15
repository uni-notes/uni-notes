# Regression

used to predict for the dependent variable on the basis of past information available on dependent and independent variables.

The estimated regression line is given by

$$
\begin{aligned}
\hat y &= b_0 + b_1 x \\
b_1 &= \frac{
	n \ \sum (xy) - \sum x \sum y
}{
	n \ \sum x^2 - \Big( \sum x \Big)^2
} \\
b_0 &= \bar y - b_1 \bar x \\
\bar x &= \frac{\sum x} n \\
\bar y &= \frac{\sum y} n
\end{aligned}
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

gives the degree of **linear** relationship between 2 vars

Properties

- Dimensionless
- Symmetric: $r(x, y)=r(y, x)$
- $r \in [-1, +1]$

$$
r(x, y) = \dfrac{\text{cov}(x, y)}{\sigma_x \sigma_y}
$$

### Pearson’s Correlation

Also called product moment correlation
$$
\begin{aligned}
r
&= \dfrac{1}{n-1} \sum_{i=1}^n z_{xi} z_{yi} \\
&= \dfrac{
	\sum (x_i - \bar x)(y_i - \bar y)
}{
\sqrt{\sum (x_i - \bar x)^2 \sum (y_i - \bar y)^2}
}
\\
&= \dfrac{
	n \sum(xy) - \sum x \sum y
}{
	n
	\sqrt{\sum (x^2) - \big(\sum x \big)^2 }
	\sqrt{ \sum (y^2) - \big(\sum y \big)^2 }
}
\end{aligned}
$$

Measures whether 2 vars are above/below mean at the same time

### Robust correlation

- Replace mean/sum with median
- Replace square with abs

$$
\begin{aligned}
r
&= \dfrac{
	\text{med} \{ \ (x_i - \tilde x)(y_i - \tilde y) \ \}
}{
\text{med} (x_i - \tilde x)
\times
\text{med} (y_i - \tilde y)
}
\end{aligned}
$$

where $\tilde x = \text{med} (x)$

### Spearman Correlation



### Modified Correlation

Setting the center as origin $\implies \bar x=\bar y=0$

- Contributes +vely if both vars are positive
- Contributes +vely if both vars are negative
- Contributes -vely if both vars are opposing sign

$$
r_0
= \dfrac{
	\sum x_i y_i
}{
\sqrt{\sum (x_i)^2 \sum (y_i)^2}
}
$$

Useful for comparing time-series, returns, etc

|               |   Type    |          Correlation          |
| :-----------: | :-------: | :---------------------------: |
| **Strength**  |   Weak    |   $\vert  r  \vert \le 0.5$   |
|               | Moderate  | $0.5 < \vert  r  \vert < 0.8$ |
|               |  Strong   |   $\vert  r  \vert \ge 0.8$   |
| **Direction** | Directly  |            $r > 0$            |
|               | Inversely |            $r < 0$            |

## Similarity to Dissimilarity

$\sqrt{2(1-r)}$

## Coefficient of Determination

$R^2$ value is used for non-linear regression. It shows how well data fits within the regression.

It has a range of $[0, 1]$. Higher the better.

$$
\begin{aligned}
R^2 &= 1 - \frac{ \text{SS}_{res} }{ \text{SS}_{tot} } \\
\text{SS}_\text{res} &= \sum\limits_{i=1}^n (y_i - \hat y)^2 \\
\text{SS}_\text{tot} &= \sum\limits_{i=1}^n (y_i - \bar y)^2 \\
\bar y &= \frac{1}{n} \sum\limits_{i=1}^n y_i
\end{aligned}
$$

where

| Symbol                 | Meaning                                                      |
| ---------------------- | ------------------------------------------------------------ |
| $\text{SS}_\text{res}$ | Residual sum of squares                                      |
| $\text{SS}_\text{tot}$ | Total sum of squares<br />Proportional to variance of the data |

