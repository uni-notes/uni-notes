Operator Method is a more general method, so it is good.

Consider a 2nd order DE

$$
\begin{align}
y'' + py' + qy &= R(x) \\(D^2 + pD + q)y &= R(x)
\end{align}
$$

## Definition

$$
y_p = \frac{1}{\phi(D)} R(x)
$$

$$
\begin{align}
\phi(D) y &= R(x) \\
\phi(D)
&= D^2 + pD + q \\&=(D-m_1)(D-m_2)
\end{align}
$$

## Integrals

$$
\begin{align}
\frac{1}{D} R(x) &= \int R(x) dx \\
\frac{1}{D^2} R(x) &= \iint R(x) dx \cdot dx
\end{align}
$$

$$
\begin{align}
\frac{1}{D-m} R(x) &= \textcolor{orange}{e^{mx}} \int R(x) \cdot \textcolor{green}{e^{-mx}} \cdot dx \\
\frac{1}{D+m} R(x) &= \textcolor{green}{e^{-mx}} \int R(x) \cdot \textcolor{orange}{e^{mx}} \cdot dx
\end{align}
$$

## Short Rules for standard functions

| R(x)                                                         | $y_p$                                                        |   Exception    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | :------------: |
| $e^{\textcolor{orange}{a}x}$                                 | $e^{\textcolor{orange}{a}x}\dfrac{1}{\phi(\textcolor{orange}{a})}$ |                |
|                                                              | $x e^{\textcolor{orange}{a}x} \frac{1}{\phi'(\textcolor{orange}{a})}$ | $\phi(a) = 0$  |
|                                                              | $x^2 e^{\textcolor{orange}{a}x} \frac{1}{\phi''(\textcolor{orange}{a})}$ | $\phi'(a) = 0$ |
| $\sin(\textcolor{orange}{a}x+b), \cos(\textcolor{orange}{a}x+b)$ | $R(x) \frac{1}{f(-\textcolor{orange}{a}^2)}$                 |                |
|                                                              | $x R(x) \frac{1}{f'(-\textcolor{orange}{a}^2)}$              | $f(-a^2) = 0$  |
|                                                              | $x^2 R(x) \frac{1}{f''(-\textcolor{orange}{a}^2)}$           | $f'(-a^2) = 0$ |
| $x^m$                                                        | $\underbrace{\Big(\phi(D) \Big)^{-1} }_\text{Binomial Series} x^m$ |                |
| $e^{\textcolor{orange}{k}x} h(x)$<br />(Exponent Shifting Rule) | $e^{\textcolor{orange}{k}x} \left\{ \frac{1}{\phi(D+\textcolor{orange}{k})} h(x) \right\}$<br />Solve using any of the above methods |                |

### Derivatives

$$
\begin{align}
\phi'(a) &= \left\{ \frac{d \phi(D)}{dD} \right\}_{D \to a} \\
\phi''(a) &= \left\{ \frac{d^2 \phi(D)}{d D^2} \right\}_{D \to a} \\f(-a^2) &= \left\{ \frac{d f(D^2)}{dD} \right\}_{D^2 \to -a^2} \\f'(-a^2) &= \left\{ \frac{d^2 f(D^2)}{d D^2} \right\}_{D^2 \to -a^2} \\
\end{align}
$$

## Binomial Expansions

$$
\begin{align}
(1+x)^{-1} &= 1 - x + x^2 - x^3 + \dots \\(1-x)^{-1} &= 1 + x + x^2 + x^3 + \dots
\end{align}
$$

$$
\begin{align}
(1+x)^{-2} &= 1 - 2x + 3x^2 - 4x^3 + \dots \\(1-x)^{-2} &= 1 + 2x + 3x^2 + 4x^3 + \dots
\end{align}
$$

$$
\begin{align}
(1+x)^{-n}
&= 1 - nx +
\frac{n(n+1) x^2}{2!} - 
\frac{n(n+1)(n+2) x^3}{3!} + \cdots\\(1-x)^{-n}
&= 1 + nx +
\frac{n(n+1) x^2}{2!} +
\frac{n(n+1)(n+2) x^3}{3!} + \cdots
\end{align}
$$

## Cube Formula

$$
(a+b)^3 =
a^3+b^3+3ab(a+b) \\
(a-b)^3 =
a^3-b^3-3ab(a-b)
$$

## Long Method

### idk

$$
\begin{align}
y_p
&= \frac{1}{\phi(D)} R(x) \\&= \underbrace{
	\left( \frac{1}{D-m_1} \right)
	\underbrace{
		\frac{1}{D-m_2} R(x)
	}_{R_1(x)}
}_{R_2(x)}\\&= \frac{1}{D-m} R(x)
\end{align}
$$

### IF

$$
\begin{align}
Dy - my &= R(x) \\
\frac{dy}{dx} - my &= R(x) \\
IF
&= e^{\int P(x) dx} \\&= e^{\int -m dx} \\&= e^{-mx}
\end{align}
$$

### Solution

$$
\begin{align}
y \times IF &= \int R(x) \cdot IF \cdot dx \\y e^{-mx} &= \int R(x) \cdot e^{-mx} \cdot dx \\y &= e^{mx} \int R(x) \cdot e^{-mx} \cdot dx
\end{align}
$$

