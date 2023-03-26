## DE

Differential equation is an equation that relates one or more unknown functions and their derivatives.

### Order

The order of a differential equation is defined to be that of the highest order derivative it contains.

### Degree

The degree of a differential equation is defined as the power to which the highest order derivative is raised.

## 1st Order DE

$$
\frac{dy}{dx} = f(x, y)
$$

Aim is to find the value of $y$ in terms in $x$. We do this by integrating (anti-derivative).

## Separable Variables

We can directly solve this by separating the variables

$$
\begin{align}
f(x, y) &= g(x) \cdot h(y) \\
\frac{dy}{dx} &= g(x) \cdot h(y) \\
\int \frac{dy}{h(y)}  &= \int g(x) dx
\end{align}
$$

## Homogeneous Equation

Special type of [Homogeneous Expression](#Homogeneous Expression)

$$
M dx + N dy = 0
$$

If both $M(x, y)$ and $N(x,y)$ are homogeneous of the same degree.

$$
\begin{align}
\frac{dy}{dx} &= \frac{- M(x, y)}{N(x, y)} \\
\text{Let }  v &= \frac{y}{x} \implies y = vx \\
\frac{dy}{dx} &= v + x \frac{dv}{dx}
\end{align}
$$

## Homogeneous Expression

$$
\begin{align}
f(tx, ty) = t^n \cdot f(x, y)
\end{align}
$$

|       Example       | Degree |
| :-----------------: | :----: |
| $\sin(\frac{x}{y})$ |   0    |
| $\sqrt{x^2 + y^2}$  |   1    |
|     $x^2 + y^2$     |   2    |

## Integration Rules

[Grade 12 Integration Rules](../../School/Math/02_Integration.md)

## Transposed Differential

Be able to identify transposition to simplify

$$
\begin{align}
d(\log x) &= \left( \frac{1}{x} \right) dx \\
\text{because
 }
\frac{d(\log x)}{dx} &= \frac{1}{x} \\
\end{align}
$$

