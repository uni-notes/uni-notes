## Chebyshev DE

$$
(1-x^2)y'' - xy' + n^2 y =0 \\n>0
\label{de}
$$

$x = \pm 1$ are the regular singular points of $\eqref{de}$

## Chebyshev Polynomials

are solutions of Chebyshev DE

Using Frobenius method near $x=1$, we get the solution 

$$
T_n = F \left( n, -n, \frac{1}{2}, \frac{1-x}{2} \right)
\label{sol}
$$

$\eqref{sol}$ is a finite polynomial of degree $n$, as $b=-n$ (-ve integer)

Using transformation $x=\cos \theta$ in $\eqref{de}$, we get another solution

$$
T_n = \cos(n \theta), \quad \theta = \cos^{-1}x

$$

## Euler’s Theorem

$$
\begin{align}
e^{i\theta}
&= \cos \theta + i \sin \theta \\
e^{ni\theta}
&= \cos n\theta + i \sin n\theta \\&= (\cos \theta + i \sin \theta)^n \\
\end{align}
$$

