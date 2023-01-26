## Functions of Several Variables

Let $D$ be the set of all $n$ tuples of the form $(x_1, x_2, \dots , x_n)$, where $x_1, x_2, \dots, x_n$ are real numbers. A function on $D$ is a rule $f$ that assigns a number $w = f(x_1, x_2, \dots, x_n)$ for each element in $D$.

If there exists only number $w$ for each element in $D$, then it is said to be a single-valued function. If more than one $w$ exists, then it is said to be a many-valued function.

While finding domain $D$

- we include all the points which make the function $f$ well-defined
- neglect the values which make $w$ a complex or undefined number

## Neighborhood

A neighborhood of a point $P_0(x_0, y_0)$ is a circular disc, with centre @ $P_0$ and radius $r$, where $r$ is a small +ve number.

If

- $r= \epsilon$, $\epsilon$ neighborhood
- $r = \delta$, $\delta$ neighborhood

In 3 dimensions, we replace circular disk with an open spherical ball with centre @ $P_0$

## Types of points

Let $S$ be a non-empty set in the XY plane . A point $P_0(x_0, y_0)$ is said to be

|  Point   | Condition                                                    |
| :------: | ------------------------------------------------------------ |
| Interior | there exists **a neighborhood** of $P_0$ which lies completely inside $S$ |
| Boundary | **every neighborhood** of $P_0$ contains points of $S$ and points outside $S$ |
| Exterior | there exists **a neighborhood** of $P_0$ completely outside $S$ |

## Types of Sets

|           | Characteristic                                       |
| :-------: | ---------------------------------------------------- |
|   Open    | contains interior points **only**                    |
|  Closed   | contains interior **and** all boundary points        |
|  Bounded  | lies completely inside an open disk of finite radius |
| Unbounded | cannot be enclosed inside open disk of finite radius |

$XY$ plane is **both** open and closed.

## Level

For a function $f(x, y)$ and constant $c$,

|               |     Equation     |
| :-----------: | :--------------: |
|  Level Curve  |  $f(x, y) = c$   |
| Level Surface | $f(x, y, z) = c$ |

## Limits

Let $f$ be a function defined at all points in the some neighborhood f $(x_0, y_0)$. We say that $f$ has a limit $L$, when the point $(x, y)$ approaches $(x_0, y_0)$ if for every $\epsilon > 0$, there exists a $\delta > 0$ such that

$$
\begin{align}
0 < \text{ Distance between } (x, y) \text{ and } (x_0, y_0) &< \delta \\0 < \sqrt{ (x-x_0)^2 + (y-y_0)^2 } &< \delta \\| f(x,y) - L | &< \epsilon \\
\implies L &= \lim_{(x, y) \to (x_0, y_0)} f(x, y)
\end{align}
$$

Here, $(x, y)$ approaches $(x_0, y_0)$ in an infinite number of ways.

### 2 Path Test

TO show that the limit of $f(x, y)$ does **not** exist @ $(x_0, y_0)$, we find 2 different paths through which the value of limits are different.

We choose the path as $y = mx^n$ or $x = m y^n$, where $m$ and $n$ are constants. The choice depends on the problem. We try to obtain a final limit in terms of $m$

## Continuity

A function $f(x, y)$ is continuous at $(x_0, y_0)$ if

1. $f(x_0, y_0)$ exists
2. $\lim_{(x, y) \to (x_0, y_0)} f(x,y)$ exists
3. $\lim_{(x, y) \to (x_0, y_0)} f(x,y) = f(x_0, y_0)$

The following functions are continuous in their domain of definition

1. Polynomial
2. Exponential
3. Circular
4. Trignometric

## Partial Derivatives

Let $f(x,y)$ be a function of 2 variables.

Provided the limit exists, the partial derivative of $f$ wrt $x$ is denoted and defined by

$$
\begin{align}
\frac{\partial f}{\partial x} &=
\lim_{\Delta x \to 0}
\frac{ f(x + \Delta x, \ y) - f(x, y) }{\Delta x} \\
\frac{\partial f}{\partial y} &=
\lim_{\Delta y \to 0}
\frac{ f(x, \ y + \Delta y) - f(x, y) }{\Delta y}
\end{align}
$$

We define higher order partial derivatives as

$$
\begin{align}
f_x &= \frac{\partial^2 f}{\partial x^2}
&= \frac{\partial}{\partial x}\left[ \frac{\partial f}{\partial x} \right] \\
f_{xy} &=\frac{\partial^2 f}{\partial x \partial y}
&= \frac{\partial}{\partial x}\left[ \frac{\partial f}{\partial y} \right] \\f_{xy} &= f_{yx} \\f_{xx} &= (f_x)_x
\end{align}
$$

## Laplace Equation

If $u$ is a function

$$
u_{xx} + u_{yy} + u_{zz} = 0
$$

## Chain Rule

If $w = f(x, y)$ a function where $x, y$ are themselves functions of

- an independent parameter $t$
  
   $$
  \frac{dw}{dt} =
  \left( \frac{\partial w}{\partial x} \cdot \frac{dx}{dt} \right) +
  \left( \frac{\partial w}{\partial y} \cdot \frac{dy}{dt} \right)
   $$

- 2 independent parameters $u, v$
  
	$$
  \begin{align}
  \frac{\partial w}{\partial u} &=
  \left( \frac{\partial w}{\partial x} \cdot \frac{\partial x}{\partial u} \right) +
  \left( \frac{\partial w}{\partial y} \cdot \frac{\partial y}{\partial u} \right) \\  
  \frac{\partial w}{\partial v} &=
  \left( \frac{\partial w}{\partial x} \cdot \frac{\partial x}{\partial v} \right) +
  \left( \frac{\partial w}{\partial y} \cdot \frac{\partial y}{\partial v} \right)
  \end{align}
   $$

## Implicit Differentiation

Let $y$ be a function of $x$, expressed as an implicit relation $f(x, y) = 0$.

Differentiating partially wrt $x$

$$
\begin{align}
\frac{\partial f}{\partial x} +
\left( \frac{\partial f}{\partial y} \cdot \frac{dy}{dx} \right)
&= 0 \\
\implies \frac{dy}{dx} &=
\frac{-\partial f / \partial x}{\partial f / \partial y} \\&= \frac{- f_x}{f_y}
\end{align}
$$

If $z$ is a function of $x$ and $y$, given by an implicit relation $f(x,y,z) = 0$

$$
\begin{align}
z_x &= \frac{-f_x}{f_z} \\z_y &= \frac{-f_y}{f_z}
\end{align}
$$

## Gradient Vector

Let $f = f(x,y)$ be a function. Then the gradient of $f$

$$
\begin{align}
\text{grad } f &= \nabla f \\&= f_x \cdot \hat i + f_y \cdot \hat j
\end{align}
$$

$\nabla$ is the vector differential operator

$\nabla f$ acts along the normal at any point to the level curve of $f$

## Directional Derivative

Let $f$ be a function defined at all pionts in some neighborhood of $P_0(x_0, y_0)$. Then, provided the limit exists, the directional derivative of $f$ in the direction of $\vec a = a_1 \hat i + a_2 \hat j$ is given by

$$
\begin{align}
\text{DD} &= (D_{\hat u} f)_{P_0} \\&= \lim_{s \to 0}
\frac{f(x_0 + su_1, y_0 + su_2) - f(x_0, y_0)}{s} \\&= \nabla f \cdot \hat u \\
\nabla f &= ( \nabla f )_{P_0} \\
\hat u &= u_1 \hat i + u_2 \hat j, \text{ unit vector in direction of } \vec a \\&= \frac{\vec A}{|\vec A|}
\end{align}
$$

### Notes

|                      Direction                      |           f            |        DD         |
| :-------------------------------------------------: | :--------------------: | :---------------: |
|                      $\nabla f$                      | increases more rapidly |  $|\nabla f|$  |
|                     $- \nabla f$                     | decreases more rapidly | $- |\nabla f|$ |
| $\perp \text{to } (\nabla f) \text{ or } (-\nabla f)$ |       no change        |         0         |

## Tangent Plane

Let $f = f(x, y, z)$. Then, the equation of the tangent plane passing through a point $P_0(x_0, y_0, z_0)$ is given by

$$
(x - x_0) {f_x}_{(P_0)} +
(y - y_0) {f_y}_{(P_0)} +
(z - z_0) {f_z}_{(P_0)}
= 0
$$

## Normal Line

The equations of normal line at $P_0$ are given by

$$
\begin{align}
x &= x_0 + t {f_x}_{(P_0)} \\y &= y_0 + t {f_y}_{(P_0)} \\z &= z_0 + t {f_z}_{(P_0)}
\end{align}, \quad t \text{ is some parameter}
$$

## Linearisation

Let $f(x, y, z)$ be a function and $P_0(x_0, y_0, z_0)$ be any point in the domain of definition. Then, the linearisation of $f$ about $P_0$ is given by

$$
L(x, y, z) = f(P_0) + (x - x_0){f_x}_{(P_0)} + (y - y_0){f_y}_{(P_0)} + (z - z_0){f_z}_{(P_0)}
$$

At all continuous points, $f$ and $L$ are the same.

## Extreme Values of a Function

Let $f(x,y)$ be a function, and $(a,b)$ be a point.

Absolute maximum is the point at which $f$ is max; absolute minimum is the point at which $f$ is minimum. They are obtained by evaluating $f$ at all local minima/maxima and comparing the values.

| Local Point | Characteristic                                               |
| ----------- | ------------------------------------------------------------ |
| Maximum     | $f(a, b) > f(x, y), \quad \forall (x, y)$ in the neighborhood of $(a,b)$ |
| Minimum     | $f(a, b) < f(x, y), \quad \forall (x, y)$ in the neighborhood of $(a,b)$ |
| Saddle      | $f$ increases in some directions and decreases in other directions at $(a, b)$ |

### Finding local points

At point $(a, b)$

$$
\begin{align}
1. & f_x = 0 \text{ and } f_y = 0 \\2. & r = f_{xx}, s = f_{xy}, t = f_{yy}, \\&D = rt - s^2 \\ 
\end{align}
$$

| $D$  | $r$  |  $(a, b)$  |
| :--: | :--: | :--------: |
| > 0  | < 0  |  Maximum   |
| > 0  | > 0  |  Minimum   |
| < 0  |    -   |   Saddle   |
| = 0  |    -   | Test Fails |

**Note:** In the above table, we can replace $r$ by $t$ as well.

## Constrained maxima, minima

We extremise a function $f(x, y, z)$ subject to constraint/condition $\phi(x, y, z) = 0$. We then proceed as follows

1. From Lagrange’s function, $\lambda =$ Lagrange’s multiplier constant
    
	  $$
    F(x, y, z) = f + \lambda \phi
    $$

2. The extreme values are given by
    
	  $$
    F_x = F_y = F_z = 0
    $$

3. Solve the equations for $x, y, z, \lambda$

**Note**

1. By Lagrange’s method, we cannot find whether $f$ has a maximum or minimum
2. If $f$ is to be extremised subject to constraints $\phi_1 = \phi_2 = 0$, then the Lagrange’s function becomes
    
	  $$
    F = f + \lambda_1 \phi_1 + \lambda_2 \phi_2
    $$
