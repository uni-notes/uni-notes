## 1D Wave Equation

Assuming that the vibration only happens in one direction.

$$
\begin{align}
a^2 &= \frac{T}{m} > 0 \\
\frac{\partial^2 y}{\partial t^2} &=
a^2 \left(
	\frac{\partial^2 y}{\partial x^2}
\right)
\end{align}

\label{idk}
$$

==$a \ne$ acceleration==

### Conditions

|                                            |                                       |        | For               |
| ------------------------------------------ | ------------------------------------- | ------ | ----------------- |
| Initial Vertical Displacement at Left End  | $y(0, t)$                             | $0$    | $\forall t$       |
| Initial Vertical Displacement at Right End | $y(\pi, t)$                           | $0$    | $\forall t$       |
| Vertical Velocity                          | $\frac{\partial y}{\partial t}(x, 0)$ | $0$    | $0 \le x \le \pi$ |
| The function                               | $y(x, 0)$                             | $f(x)$ | $0 \le x \le \pi$ |

### Solution

Solution of $\eqref{idk}$ under the initial conditions

$$
\begin{align}
y(x, t)
&= \sum_{n = 1}^\infty
b_n
\sin(nx)
\textcolor{hotpink}{\cos (nat)} \\
b_n &= \frac{2}{\pi} \int\limits_0^\pi f(x) \sin(nx) dx
\end{align}
$$

## 1D Heat Equation

### Fourier Thermal Law

> The amount of heat flowing through a heat-producing body $H$
>
> - $H \propto$ temperature gradient
> - $H \propto$ area of cross-section
> - $H \frac{1}{\propto}$ resistance

Time 0 is the time at which the external temperature is placed

### Formula

$\alpha^2$ is thermal diffusability.

$$
\begin{align}
\alpha^2
&= \frac{k}{\rho c} > 0\\
\frac{\partial u}{\partial t}
&=
\alpha^2 \left( \frac{\partial^2 u}{\partial x^2} \right)
\end{align}

\label{2}
$$

### Conditions

|                           |             |        | For               |
| ------------------------- | ----------- | ------ | ----------------- |
| Initial Heat at Left End  | $u(0, t)$   | 0      | $\forall t$       |
| Initial Heat at Right End | $u(\pi, t)$ | 0      | $\forall t$       |
|                           | $u(x, 0)$   | $f(x)$ | $0 \le x \le \pi$ |

### Solution

Solution of $\eqref{2}$ under the initial conditions

$$
\begin{align}
u(x, t)
&= \sum_{n = 1}^\infty
b_n
\sin (nx)
\textcolor{hotpink}{e^{-n^2 \alpha^2 t}} \\
b_n &= \frac{2}{\pi} \int\limits_0^\pi f(x) \sin(nx) dx
\end{align}
$$
