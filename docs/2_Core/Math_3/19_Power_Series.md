## Power Series

An infinite series in $x$ of the form

$$
\sum_{n=0}^\infty a_n x^n = a_0 + a_1 x + a_2 x^2 + \dots + a_r x^r + \dots
$$

where $\{ a_0, a_1, a_2, \dots \}$ are constants

equation is convergent only when $x \to 0$

### Non-Algebraic Elementary Functions

Transcendental means non-algebraic

| Function     | Power Series                                                | Intuition                     |
| ------------ | ----------------------------------------------------------- | ----------------------------- |
| $e^x$        | $1 + \frac{x}{1!} + \frac{x^2}{2!} + \dots$                 |                               |
| $\cos x$     | $1 - \frac{x^2}{2!} + \frac{x^4}{4!} + \dots$               | Even function so even numbers |
| $\sin x$     | $x - \frac{x^3}{3!} + \frac{x^5}{5!} + \dots$               | Odd function so odd numbers   |
| $\cosh x$    | $1 + \frac{x^2}{2!} + \frac{x^4}{4!} + \dots$               |                               |
| $\sinh x$    | $x + \frac{x^3}{3!} + \frac{x^5}{5!} + \dots$               |                               |
| $\log(1+x)$  | $x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \dots$ |                               |
| $(1+x)^{-1}$ | $1 - x + x^2 - x^3 + \dots$                                 |                               |
| $(1-x)^{-1}$ | $1 + x + x^2 + x^3 + \dots$                                 |                               |

## Solving

$$
\begin{aligned}
y
&= \sum_{n=0}^\infty a_n x^n
&&= a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \dots \\
y'
&= \sum_{n=1}^\infty a_n n x^{n-1}
&&= a_1 + 2a_2 x + 3a_3 x^2 + \dots \\
y''
&= \sum_{n=2}^\infty a_n n(n-1) x^{n-2}
&&= 2a_2 + (3 \cdot 2) a_3 x + \dots \\
\end{aligned}
$$

### Comparing Coefficients

By changing index and re-arranging terms, we have to make the following equal

1. counter start
2. $x$ power

## 2nd Order

Power series solution is only possible if $x = 0$ is an [ordinary point](#ordinary point) of the DE.

## Types of Points

Consider a general 2nd order differential equation with polynomials $P_1, P_2, P_3$.

$$
\begin{aligned}
P y'' + Q y' + R y &= 0 \\
\implies y'' + \frac{Q}{P} y' + \frac{R}{P} y &= 0
\end{aligned}
$$

| Types          | $P(a)$  |
| -------------- | ------- |
| Ordinary Point | $\ne 0$ |
| Singular Point | $= 0$   |

## Ordinary Point

$x=a$ is an ordinary point of DE equation, if $P(a) \ne 0$.

### Power Series Solution

The power series solution of equation is given by

$$
y = \sum_{n=0}^{\infty} a_n x^n
$$

### General Solution

Solving equation as

$$
y = a(\text{PS}_1) + b(\text{PS}_2)
$$

where

- PS~1~ and PS~2~ are congruent and linearly-independent power series, for $x \to 0$
- $a$ and $b$ are arbitrary constants

## Singular Points

Consider limits

$$
\begin{aligned}
p&=
\lim_{x \to a} (x-a)
&\frac{Q(x)}{P(x)}\\
q&=
\lim_{x \to a} (x-a)^{\textcolor{hotpink}{2}}
&\frac{R(x)}{P(x)}
\end{aligned}
$$

| Both limits exist | Point Type |
|:-:|:-:|
|    ✅    |     Regular     |
|    ❌    |			Irregular |

### Frobenius Series Method

Differential equations with **regular** singular points at $x=0$ can be solved using a power series of the form

$$
\begin{aligned}
y
&= x^m \sum_{n=0}^\infty a_n x^n \\&= \sum_{n=0}^\infty a_n x^{m+n}
\end{aligned}
$$

where $m$ is constant coefficient called as root/indical/initial value. This is singular points (to be calculated).

### Trick to find indical value

$$
m(m-1) + p m + q = 0
$$

where $p$ and $q$ are limits from equation
