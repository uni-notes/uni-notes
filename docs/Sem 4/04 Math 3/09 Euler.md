## Euler’s Equidimensional DE

$$
x^2 y'' + px y' + qy = 0 \label{gen}
$$

## Transformation

1. Let $x = e^z \quad (z = \log x)$

1. Now, $y$ is a function of $z$, which in turn is a function of $x$

1. Put the following substitutions; Refer to [Custom Operators](#Custom Operators)
   

$$
   \begin{align}
   xD &= \theta \\   x^2 D^2 &= \theta(\theta - 1)
   \end{align}
   

$$
   
1. Equation $\eqref{gen}$ becomes
   

$$
   \begin{align}
   \Big( \theta(\theta - 1) + p \theta + q \Big)y &= 0 \\   \theta(\theta - 1) + p \theta + q &= 0 & (y \ne 0)
   \end{align}
   

$$
   
1. Put $\theta^2 \to m^2, \theta \to m$

1. Find gen solution in terms of $z : y(z)$, using [Constant Coefficient](08 Constant Coefficient.md)

     - $y = c_1 e^{m_1 z} + c_2 e^{m_2 z}$
     - $y = e^{mz}(c_1 + c_2 z)$
     - $y = e^{az}(c_1 \cos bz+ c_2 \sin bz)$

1. Find gen solution in terms of $x$, by subbing $z = \log x$

## Custom Operators

$$
\begin{align}
D &= \frac{d}{dx}  &
D^2 &= \frac{d^2}{dx^2} \\
\theta &= \frac{d}{dz}  &
\theta^2 &= \frac{d^2}{dz^2}
\end{align}
$$

### Formula

$$
x Dy = \theta y \implies xD = \theta

$$

