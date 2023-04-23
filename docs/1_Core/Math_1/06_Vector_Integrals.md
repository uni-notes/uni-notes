## Line Integrals

Let $f(x, y, z)$ be a function whose domain consists of a smooth curve $C: \vec r(t) = x(t) \hat i + y(t) \hat j + z(t) \vec k$. Then, then line integral of $f$ over $C$ is given by

$$
\int\limits_C f(x, y, z) \ ds = \int\limits_C f(x, y, z) \cdot |\vec V| \ dt
$$

because displacement s = $\int$ velocity = $\int$ speed x direction

**Note**

1. We aevaluate the integral by converting the integral in terms of a parameter $t$, or writing in terms of any one variable $x$ or $y$ or $z$ alone
2. A curve is smooth if $\frac{d \vec r}{dt} \ne 0$ and $\frac{d \vec r}{dt}$ is a constant
3. A closed curve which doesn’t cross itself is called a simple closed curve
4. If $C$ is a simple closed curve enclosing a region $R$, then +ve direction is that direction through which one walks such that the enclosed region on their **left**

## Work Done

The work done by a force field $\vec F = M \hat i + N \hat j + P \vec k$ along curve $C: x \hat i + y \hat j + z \hat k, a \le t \le b$ is

$$
\begin{aligned}
W &= \int\limits_C \vec F \cdot d \vec r \\
&= \int\limits_C (M \ dx + N \ dy + P \ dz) \ dt
\end{aligned}
$$

The above integral is also referred to as the **circulation of vector $\vec F$** in fluid flow problems.

## Conservative Forced Field

If the line integral is independent of the path of integration, then $\vec F$ is said to conservative/irrotational.

A force $\vec F = M \hat i + N \hat j + P \vec k$ is conservative

1. $$
   \begin{aligned}
   M_y &= N_x \\
	 P_y &= N_z \\
	 P_x &= M_z
   \end{aligned}
   $$

2. there exists a scalar potential function $\phi(x, y, z)$ such that
   
$$
\begin{aligned}
\vec F &= \nabla \phi \\
\text{where } \nabla \phi &= \phi_x \hat i + \phi_y \hat j + \phi_z \hat k
\end{aligned}
$$

3. If $C$ is any path joining A and B

$$
\begin{aligned}
W &= \int\limits_C \vec F \cdot d \vec r \\   &= \phi(B) - \phi(A)
\end{aligned}
$$

## Green’s Theorem in a Plane

Let $\vec F = M \hat i + N \hat j$ be a vector-valued function defined at all points in a region $R$ in the $XY$ plane, bounded by a simple closed curve C. Then, the counter-clockwise circulation of $\vec F$ or flux or tangential form of Green’s theorem is given by

$$
\begin{aligned}
\oint\limits_C \vec F \cdot d \vec r &= \int\limits_C M \ dx + N \ dy \\
&= \iint\limits_R 
\left( \frac{\partial N}{\partial x} - \frac{\partial M}{\partial y} \right)
\ dx \ dy
\end{aligned}
$$

## Gauss Divergence Theorem

Let $F = M \hat i + N \hat j + P \hat k$ be a vector-valued function, defined at all points of closed surface $S$, enclosing a volume $V$. Then, the outward-drawn flux of $\vec F$ is given by

$$
\begin{aligned}
\iint_S \vec F \cdot \vec n \cdot ds &=
\iiint (\text{div } \vec F) \ dv \\
\text{where }
(\text{div } \vec F) &= \nabla \cdot \vec F \\
&= \frac{\partial M}{\partial x} + \frac{\partial N}{\partial y} + \frac{\partial P}{\partial z} \\
\vec n &= \frac{\nabla \phi}{ |\nabla \phi| } \\
&\text{(unit outward-drawn normal vector to surface S)}\\
\phi &= \phi(x, y, z) \\
&\text{(equation of surface S)}
\end{aligned}
$$

## Stoke’s Theorem

If $\vec F = M \hat i + N \hat j + P \vec k$ is defined on all points on an open surface bounded by a simple curve $C$,

$$
\begin{aligned}
\int \limits_C \vec F \cdot dr &=
\iint \limits_S (\text{curl } \vec F) \cdot \hat n \cdot ds \\
\text{where }
(\text{curl } \vec F) &= \vec V \times \vec F \\
&= \begin{vmatrix}
\hat i & \hat j & \hat k \\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\
M & N & P
\end{vmatrix}
\end{aligned}
$$

