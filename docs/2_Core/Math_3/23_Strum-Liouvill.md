Consider the DE with scalar $\lambda$ defined in $[a,b]$

$$
\frac{d}{dx}
\Big[
	P(x) y'
\Big]
+
\Big[\lambda Q(x) + R(x) \Big] y
= 0
$$

with the boundary conditions

$$
\begin{aligned}
c_1 y(a) + c_2 y'(a) &= 0 &
d_1 y(b) + d_2 y'(b) &= 0 \\
c_1 \text{ or } c_2 &= 0 &
d_1 \text{ or } d_2 &= 0
\end{aligned}
$$

### Simplest Form

$$
\begin{aligned}
y'' + \lambda y &= 0 \\
P(x) &= 1 \\
Q(x) &= 1 \\
R(x) &= 0
\end{aligned}
$$

### Legendre Equation

Legendre Equation can be represented as Strum-Liouvile Problem.

$$
\frac{d}{dx}
\Big[
	\underbrace{(1-x^2)}_{P(x)}
	y'
\Big] +
\underbrace{n(n+1)}_{\lambda} \ y
= 0 \\
P(x) = 1-x^2 \\
Q(x) = 1 \\ R(x) = 0 \\
\lambda = n(n+1)
$$

Here, $\lambda$ is the eigen value of equation

The corresponding solutions are $P_n(x), n = 1, 2, \dots$ They are called as eigen functions.

$n > 0$ because $n \le 0$ will give trivial solution.

## Eigen Value/Function

$$
\begin{aligned}
y'' + \lambda y &= 0 \\
y(a) &= 0 \\
y(b) &= 0 \\
a & \ne b
\end{aligned}
$$

