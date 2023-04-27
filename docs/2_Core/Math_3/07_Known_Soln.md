Consider a homogeneous 2nd order DE.

$$
y'' + P y' + Q y = 0
$$

Let $y_1(x)$ be the known solution of it.

To find another **linear-independent** solution $y_2(x)$

1. Let
   
$$
\begin{aligned}
v &= \int
\frac{1}{ {(y_1)}^2 \times e^{ \int P dx} } \\
y_2 &= v \cdot y_1
\end{aligned}
$$
   
2. Now, the general solution $y(x) = c_1 y_1(x) + c_2 y_2(x)$

## Special Cases

(not important)

### Legendre DE

$$
(1-x^2)y'' - 2xy' + k(k+1) y = 0
$$

where $k$ = const

### Bessel’s Equation

$$
x^2 y'' + xy' + (x^2 - k^2) y = 0
$$

$k$ = const

