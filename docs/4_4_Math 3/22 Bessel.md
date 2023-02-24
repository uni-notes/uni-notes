## Bessel’s DE

Family of differential equation, with some constant value $p$

$$
x^2y'' + xy' + (x^2-p^2) y = 0
\label{de}
$$

## Bessel’s Function

is the solution of Bessel’s DE. Denoted by $J_p(x)$

$x=0$ is a regular singular point of $\eqref{de}$. Solving using Frobenieus Series method gives 2 initial roots as $m = \pm p$

|        |                             $+p$                             |                             $-p$                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| $J(x)$ | $\sum\limits_{n=0}^\infty \dfrac{(-1)^n \left(\frac{x}{2}\right)^{2n \textcolor{hotpink}{+p}}}{n!(n \textcolor{hotpink}{+p})!}$ | $\sum\limits_{n=0}^\infty \dfrac{(-1)^n \left(\frac{x}{2}\right)^{2n \textcolor{hotpink}{-p} }}{n!(n \textcolor{hotpink}{-p} )!}$ |

The above 2 formula are not directly possible for negative integers, as $(n-p)!$ is not valid when it is negative<br />Use [gamma function](14 Laplace.md#Gamma Function)

## General Solution

$$
y = c_1 J_p(x) + c_2 J_{-p} (x)
$$

## Properties

### To Remember

$$
\begin{align}
J_\frac{1}{2}(x) &= \sin x \sqrt{
	\frac{2}{\pi x}
} \\
J_\frac{-1}{2}(x) &= \cos x \sqrt{
	\frac{2}{\pi x}
} \\
J_{p-1}(x) + J_{p+1}(x) &= \frac{2p}{x} J_p(x)
\end{align}
$$

### Other Properties

$$
\begin{align}
\Big( x^{p} J_p(x) \Big)'
&= x^{p} J_{p-1} (x) \\
\Big( x^{-p} J_p(x) \Big)'
&= - x^{-p} J_{p+1} (x) \\
{J_p}'(x) + \frac{p}{x} J_p(x) &= J_{p-1}(x) \\{J_p}'(x) - \frac{p}{x} J_p(x) &= - J_{p+1}(x) \\
J_{p-1}(x) - J_{p+1}(x) &= 2 {J_p}'(x)
\end{align}
$$

