## Polar Coordinates

In polar coordinate system, we locate a point with reference to:

1. **pole** a fixed point
   (usually fixed at the origin)
2. **initial ray** a fixed line, passing through the pole
   (usually $+x$ axis)

Let 

- $r$ - **directed** distance of the point from pole
    - $r > 0$ forward
    - $r < 0$ backward
- $\theta$ - **directed** angle of radius vector from the initial ray
    - $\theta < 0$ anti-clockwise
    - $\theta > 0$ clockwise
- $P(r, \theta)$ - corresponding point

![Polar](img/polar.svg)

## Circle Through Pole

$$
r = \pm a, \quad 0 \le \theta \le 2 \pi
$$

represents a circle with center @pole and radius $a$. Sign can be either, because it is the same circle traversed in the opposite direction

## Straight line through pole

$$
\theta = \theta_0, \quad - \infty < r < \infty
$$

## IDK

| $r$        | $\theta$   | Diagram               |
| ---------- | ---------- | --------------------- |
| const      | const      | point                 |
| const      | inequality | arc                   |
| inequality | const      | straight line segment |
| inequality | inequality | region                |

## Cartesian $\iff$ Polar

Consider the point $P(x, y) \iff P(r, \theta)$

$$
\begin{align}
x &= r \cos\theta \\y &= r \sin\theta \\
r^2 &= x^2 + y^2 \\
\theta &= \tan^{-1} \left( \frac y x \right)
\end{align}
$$

## Symmetry

Let $r = f(\theta)$ be a polar curve

### X-axis

$P(r, \theta)$ and $P'(r, - \theta)$ lie on same graph

| Symmetry about | Vary theta                                  | $P(r, \theta)$ lies on the same graph as | or $P(r, \theta)$ lies on the same graph as |
| -------------- | ------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| X-axis         | $0 \le \theta \le \pi$                      | $P'(r, -\theta)$                         | $P'(-r, \pi -\theta)$                       |
| Y-axis         | $\frac{-\pi} 2 \le \theta \le \frac \pi 2$ | $P'(-r, -\theta)$                        | $P'(r, \pi -\theta)$                        |
| Origin         | $0 \le \theta \le \frac \pi 2$              | $P'(-r, \theta)$                         | $P'(r, \pi + \theta)$                       |

## Shapes

### Limacon

$$
r = a \pm b \cos\theta \\
\text{ or } \\r = a \pm b \sin\theta
$$

| $\frac a b$ | Type       |
| ----------- | ---------- |
| $<1$        | inner loop |
| $=1$        | cardioid   |
| $>1$        | outer loop |

### Roses

$$
\begin{align}
r &= a \cos(n\theta) \\&\text{ or } \\r &= a \sin(n\theta) \\
\text{No of petals } N &= \begin{cases}
n, &  n = \text{odd} \\2n, & n = \text{even}
\end{cases} \\
\text{Axis of first petal } \theta &= 
\begin{cases}
0 &  r = a \textcolor{red}{\cos}(n \theta) \\
\dfrac \pi {2n} & r = a \textcolor{red}{\sin} (n \theta)
\end{cases} \\
\text{Length of petals} &= a \\
\text{Angular Gap between axes of petals} &= \frac{2 \pi}{N}
\end{align}
$$

### Lemmiscates

$$
r^2 = a \cos\theta \\
\text{ or } \\r^2 = a \sin\theta \\
$$

### Straight Line

$$
r \cos(\theta-\theta_0) = r_0
$$

- $P(r, \theta)$ is any point on given line
- $P_0(r_0, \theta_0)$ is foot of $\perp$r from the pole

### Circle

$$
r^2 + {r_0}^2 - 2 r r_0 \cos(\theta - \theta_0) = a^2 \\
$$

- $P(r, \theta)$ is any point on circle
- $P_0(r_0, \theta_0)$ is center of circle
- $a$ is radius

#### Radius passing through pole

$$
r_0 = a\\r = 2a cos(\theta - \theta_0)
$$

#### Center lies on axis

|      Center at      |        $r$        |
| :-----------------: | :---------------: |
|       $(a,0)$       | $2a \cos \theta$  |
|      $(-a,0)$       | $-2a \cos \theta$ |
| $(a, \frac \pi 2)$  | $2a \sin \theta$  |
| $(a, -\frac \pi 2)$ | $-2a \sin \theta$ |

## Area under curve

For a polar curve $r = f(\theta), \alpha \le \theta \le \beta$

$$
A = \frac12 \int\limits_{\theta = \alpha}^\beta
r^2
\cdot d\theta
$$

For area bounded by the curves $r_1 = f_1(\theta), r_2 = f_2(\theta), \alpha \le \theta \le \beta$ such that $r_1 < r_2$

$$
A = \frac12 \int\limits_{\theta = \alpha}^\beta
{r_2}^2 - {r_1}^2
\cdot d\theta
$$

## Length of curve

For a curve $r = f(\theta), \alpha \le \theta \le \beta$ traversed **exactly once** from $\theta = \alpha \to \beta$

$$
L =
\int\limits_{\theta = \alpha}^\beta
\sqrt{ r^2 + (r')^2 }
\cdot d\theta \qquad
\left[ r' = \frac{dr}{d \theta} \right]
$$

## Conic Sections

Let

- $P(r, \theta)$ be any point on the conic section with focus at origin
- $e = \dfrac{ \text{Distance bw focii} }{ \text{Distance bw vertices} }$

| Directrix |              $r$               |
| :-------: | :----------------------------: |
|  $x = a$  | $\frac{ke}{1 + e \cos\theta}$ |
| $x = -a$  | $\frac{ke}{1 - e \cos\theta}$ |
|  $y = a$  | $\frac{ke}{1 + e \sin\theta}$ |
| $y = -a$  | $\frac{ke}{1 - e \sin\theta}$ |

### Shapes

|     $e$     | Shape     |
| :---------: | --------- |
| $0 < e < 1$ | Ellipse   |
|   $e = 1$   | Parabola  |
|   $e > 1$   | Hyperbola |

For ellipse,

$$
k = a \left[ \frac 1 e - e \right]
$$