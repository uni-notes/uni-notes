## Legendre DE

$$
(1-x^2) y'' -
2xy' +
n(n+1)y
= 0
$$

## Solution

Solution of equation, at the 2 singular points $x = \pm 1$.

We will get

$$
t(t-1)y'' + (1-2t)y' + n(n+1)y = 0
$$

equation is a hyper-geometric function.

### Solution of equation near $t=0$

$$
y = F(-n, n+1, 1, t)
$$

### Solution of equation near $x=0$

$$
P_n(x) =
y =
F \left(-n, n+1, 1, \frac{1-x}{2} \right)
$$

This is a legendre polynomial of degree $n$.

## Rodrigue’s Formula

$$
P_n(x) =
\frac{1}{2^n \cdot n!}
\left[
	\frac{d^n}{d x^n} (x^2 - 1)^n
\right]
$$

$$
\begin{aligned}
P_0(x) &= 1 \\P_1(x) &= x \\P_2(x) &= \frac{1}{2}(3x^2 - 1) \\P_3(x) &= \frac{1}{2}(5x^3 - 3x)
\end{aligned}
$$

## Generating Function

$$
(1-2xt + t^2)^{-1/2} = \sum\limits_{n=0}^\infty P_n (x) t^n \\
|t| < 1 \\
|x| \le 1
$$

## Binomial Expansion

$$
(1+t^2)^{-1/2} =
$$

## Legendre Series

Similar to [Fourier Series](17_Fourier_Series.md), Any function $f(x)$ can be represented as

$$
\begin{aligned}
f(x) &= \sum_{n=0}^\infty a_n P_n(x) \\
a_n &= \frac{2n+1}{2} \int_{-1}^1 f(x) P_n(x) dx
\end{aligned}
$$

