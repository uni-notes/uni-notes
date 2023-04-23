## Vector Valued Functions

The motion of a particle moving space is given by

$$
\vec r = x(t) \cdot \hat i + y(t) \cdot \hat j + z(t) \cdot \hat k, \\
a \le t \le b, \quad a, b \in R
$$

## Limits

$\vec r(t)$ has a limit $\vec L$ as $t$ approaches $t_0$ if the following is satisfied
For every $\epsilon > 0$, there exists a $\delta > 0$ such that $0<|t - t_0|< \delta \implies | \vec r(t) - \vec L | < \epsilon$

The limit is denoted as

$$
\lim_{t \to t_0} \vec r(t) = \vec L
$$

## Continuity

$r(t)$ is continuous @ $t = t_0$ if

1. $\vec r(t_0)$ exists
2. $\lim_{t \to t_0} \vec r(t)$ exists
3. $\lim_{t \to t_0} \vec r(t) = \vec r(t_0)$

## Derivative

$$
\frac{dr}{dt} =
\lim_{\Delta t \to 0} \frac{
	\vec r(t + \Delta t) - \vec r(t)
}{\Delta t}
$$

| Quantity     |                                |                             |
| ------------ | ------------------------------ | --------------------------- |
| Velocity     | $\frac{d \vec r}{d t}$        |                             |
| Acceleration | $\frac{d \vec V}{d t}$        | $\frac{d^2 \vec r}{d t^2}$ |
| Speed        | $\|\vec V\|$                 |                             |
| Direction    | $\frac{\vec V}{\|\vec V\|}$ |                             |

### Note

Velocity = Speed $\times$ Direction

The path of a particle is said to be smooth if

1. $\frac{d \vec r}{d t} \ne 0$
2. $\frac{d \vec r}{d t}$ is continuous

If $\vec u$ is a vector of constant length, then $\vec u \cdot \frac{d \vec u}{d t} = 0$
(circle, perpendicular, cos 90 = 0)

The path of a particle is gievn by eliminating the parameter $t$ from $x, y, z$
eg: The path of a particle having $\vec r(t) = \cos t \cdot \hat i + \sin t \hat j, \quad t \in I$

$$
\begin{aligned}
x^2 + y^2 &=
\cos^2 t + \sin^2 t \\
&= 1
\end{aligned}
$$

Therefore, this path is a circle with radius = 1

## Angle Between Vectors

$$
\begin{aligned}
\cos \theta &=
\frac{
	\vec a \cdot \vec b
}{
	|\vec a| |\vec b|
} \\
\theta &= \cos^{-1} \left(
\frac{
	\vec a \cdot \vec b
}{
	|\vec a| |\vec b|
}
\right)
\end{aligned}
$$

## Arc Length

If

- $\vec r(t)$ is a smooth curve, traversed **exactly once** from $t=a \to b$
- $\vec V$ is the velocity vector

$$
L = \int\limits_a^b 
|\vec V(t)| \cdot dt
$$

Length is basically the integral of speed

## Arc Length Parameter

If $\vec r(t) \quad t \ge t_0$ is a smooth curve, then arc length parameter wrt base point @ $t=0$ is

$$
L = \int\limits_{\tau = t_0}^t 
|\vec V(\tau)| \cdot d \tau
$$

## Special Vectors

| Vector | Symbol |                                                              |                                 |
| :--------------------------- | :------: | :----------------------------------------------------------: | :-----------------------------: |
| Unit Tangent Vector          | $\hat T$ | $\frac{ \frac{d \vec r}{dt} }{ \|\frac{d \vec r}{dt}\| }$ | $\frac{\vec V}{\| \vec V \|}$ |
| Principle Unit Normal Vector | $\hat N$ | $\frac{ \frac{d \vec T}{dt} }{ \|\frac{d \vec T}{dt}\|}$ |                                 |
| Curvature <br />Rate of change in direction of curve, wrt arc length |   $k$    |                    $\frac{d \vec T}{d s}$                    | $\frac{1}{\| \vec V \|} \cdot \|\frac{d \hat T} {dt}\|$ |
| Radius of Curvature | $\rho$ | $\frac{1}{k}$ |  |

Curvature @ any point on a

- straight line is 0
- smaller circle will be greater than that of a larger one

## Components of Vector

If $\vec a = a_t \cdot \hat T + a_N \cdot \hat N$, then

| Component  | Symbol |                        |                                    |
| ---------- | :----: | :--------------------: | :--------------------------------: |
| Tangential | $a_T$  | $\frac{d \|V \|}{dt}$ |                                    |
| Normal     | $a_N$  |     $k \|V\|^2$      | $\sqrt{ \|\vec a\|^2 - {a_T}^2}$ |

### Note

If speed is contant

- $a_T = 0$
- all acceleration wil be in direction of $\hat N$

$a_T$ only exists when objects speed up / slow down

