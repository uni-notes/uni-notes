## Types

### Complete Equation

$$
y'' + P(x) y' + Q(x) y = R(x)
$$

Also called non-homogeneous DE
Particular Solution of complete equation: $y_p(x)$
If $y(x)$ is the solution, then it is given by

$$
y(x) = y_g + y_p
$$

### Reduced Equation 

Complete equation with $R(x) = 0$

$$
y'' + P(x) y' + Q(x) y = 0
$$

Also called as homogeneous DE

$$
y(x) = y_g \quad (y_p(x) = 0)
$$

## Theorems

### 1

If $y_1(x)$ and $y_2(x)$ are 2 solutions of reduced DE, then $\set{c_1 y_1(x) + c_2 y_2(x)}$ is another solution of the reduced DE for any constants $c_1, c_2$

### 2

If $y_1(x)$ and $y_2(x)$ are 2 solutions of reduced DE, then they are
**linearly-dependent** $\iff$ their **wronskian** = 0

$$
W(y_1, y_2) = 
\begin{vmatrix}
y_1 & y_2 \\
{y_1}' & {y_2}'
\end{vmatrix}
= 0
$$

Else, they are linearly-independent

eg:

- $y_1 = x^2, y_2 = \frac{3}{2} x^2$ - linear dependent
- $y_1 = x, y_2 = x^2$ - linearly independent

### 3

If $y_1(x)$ and $y_2(x)$ are 2 **linearly-independent** solutions of reduced DE, then $y(x) = c_1 y_1(x) + c_2 y_2(x)$ is called general solution

## Solving

1. Sub $y = y_1(x)$ and $y = y_2(x)$ in the given equation
2. Show that LHS = RHS

## Principle of Superposition

If the given DE is of the form

$$
y'' +  py' + qy = f(x) + g(x)
$$

Solution is given by

$$
\begin{align}
y'' +  py' + qy &= 0 &\to y_g \\
y'' +  py' + qy &= f(x) &\to y_{p_1} \\
y'' +  py' + qy &= g(x) &\to y_{p_2} \\
\implies y &= y_g + y_{p_1} + y_{p_2}
\end{align}
$$

This superposition of the solutions is called as principle of superposition.
