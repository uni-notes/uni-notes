

$$
\begin{align}
f(x, y)
&= P(X=x, Y=y) \\&= P(x \cap y) \\F(x,y) &= P(X \le x, Y \le y) \\
f(x, y) &\ge 0 \\f(x|y) &= \frac{f(x, y)}{f(y)}
\end{align}
$$

|           | Discrete                                       | Continuous                                                   |
| --------- | ---------------------------------------------- | ------------------------------------------------------------ |
| PDF       | $\sum\limits_x \sum\limits_y f(x, y) = 1$      | $\int\limits_x \int\limits_y f(x, y) \ \mathrm{d} y \mathrm{d} x = 1$      |
| CDF       | $\sum\limits_0^x \sum\limits_0^y f(x, y)$      | $\int\limits_{- \infty}^x \int\limits_{- \infty}^y f(x, y) \ \mathrm{d} y \mathrm{d} x$ |
| $f(x)$    | $\sum\limits_y f(x,y)$                         | $f(x) = \int\limits_y f(x,y) \ \mathrm{d} y \mathrm{d} x$                  |
| $f(y)$    | $\sum\limits_x f(x,y)$                         | $\int\limits_x f(x,y) \ \mathrm{d} y \mathrm{d} x$                         |
| $E(x, y)$ | $\sum\limits_x \sum\limits_y xy \cdot f(x, y)$ | $\int\limits_x \int\limits_y xy \cdot f(x, y) \ \mathrm{d} y \mathrm{d} x$ |
| $E(x)$    | $\sum\limits_x x \cdot f(x,y)$                 | $\int\limits_x x \cdot f(x,y) \ \mathrm{d} y \mathrm{d} x$                 |
| $E(y)$    | $\sum\limits_y y \cdot f(x,y)$                 | $\int\limits_y y \cdot f(x,y) \ \mathrm{d} y \mathrm{d} x$                 |

## Covariance

$$
\begin{align}
\text{Cov} (x,y) &= E(x,y) - E(x) \cdot E(y) \\&= 
\begin{cases}
>0 & \text{directly-dependent} \\0 & \text{independent}\\<0 & \text{inversely-dependent}
\end{cases}
\end{align}
$$

## Independence

$$
\begin{align}
f(x,y) &= f(x) \cdot f(y) \\E(x, y) &= E(x) \cdot E(y) \\
\text{Cov}(x,y) &= 0
\end{align}
$$

