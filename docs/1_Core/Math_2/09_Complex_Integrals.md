## Line Integral

For $\int f(z) \ dz$, put $z = r \cdot e^{i \theta}$

## ML Inequality

maximum value / upper bound of integral

$$
\left| \int_C f(z) \ dz \right|
\le M \times L
$$

where

- $M =$ max value of $f(z)$
- $L =$ length of contour $C$

## Theorems

| Theorem                 |          Cauchy-Goursat          |                       Cauchy-Integral                        |               Cauchy-Integral for derivatives                |                        Cauchy Residue                        |
| ----------------------- | :------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Condition               | $f(z)$ is analytic inside/on $C$ | - $f(z)$ is analytic inside/on $C$<br />- $z_0$ is a point inside $C$ | - $f(z)$ is analytic inside/on $C$<br />- $z_0$ is a point inside $C$ |                                                              |
| Identity                |      $\int_C f(z) \ dz = 0$      |    $\int_C \frac{f(z)}{z-z_0} dz = 2 \pi i \cdot f(z_0)$     | $\int_C \frac{f(z)}{(z-z_0)^{n+1}} dz = \frac{2 \pi i}{n!} \times f^{(n)}(z_0)$ | $\int_C f(z) \ dz = 2 \pi i \times \\ [\text{Sum of residues at poles lying inside/on } C]$ |
| add for multiple points |                ❌                 |                              ✅                               |                              ✅                               |                                                              |

| Theorem                             |                          Condition                           |                           Identity                           | add for multiple points |
| ----------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------: |
| **Cauchy-Goursat**                  |               $f(z)$ is analytic inside/on $C$               |                    $\int_C f(z) \ dz = 0$                    |            ❌            |
| **Cauchy-Integral**                 | - $f(z)$ is analytic inside/on $C$<br />- $z_0$ is a point inside $C$ |    $\int_C \frac{f(z)}{z-z_0} dz = 2 \pi i \cdot f(z_0)$     |            ✅            |
| **Cauchy-Integral for derivatives** | - $f(z)$ is analytic inside/on $C$<br />- $z_0$ is a point inside $C$ | $\int_C \frac{f(z)}{(z-z_0)^{n+1}} dz = \frac{2 \pi i}{n!} \times f^{(n)}(z_0)$ |            ✅            |
| **Cauchy Residue**                  |                                                              |         $\int_C f(z) \ dz = 2 \pi i \times (\sum R)$         |            ❌            |

$\sum R =$ Sum of residues at poles lying inside/on $C$

### Residue

|                             Type                             |                             $R$                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         Simple Pole                          |               $\lim_{z \to z_0} (z-z_0) f(z)$                |
|                      Pole of order $m$                       | $\dfrac{1}{m-1} \times \dfrac{d^{m-1}}{dz^{m-1}} [(z-z_0)^m f(z)]_{z = z_0}$ |
| $\dfrac{P(z_0)}{\textcolor{orange}{Q}(z_0)}, P(z_0) \ne 0, Q(z_0) = 0$ |        $\dfrac{P(z_0)}{\textcolor{orange}{Q'}(z_0)}$         |

## Laurent’s Series

$$
\begin{aligned}
f(z) &= \sum_0^\infty a_n (z-z_0)^n + \underbrace{
	\sum_1^\infty \frac{b_n}{(z - z_0)^n}
}_\text{Principal Part} \\a_n &= \frac{1}{2 \pi i} \times \int \frac{f(z)}{(z-z_0)^{
	\textcolor{orange}{n}+1
}} \\b_n &= \frac{1}{2 \pi i} \times \int \frac{f(z)}{(z-z_0)^{
	\textcolor{orange}{-n}+1
}} 
\end{aligned}
$$

The following equation $\eqref{formula}$ is only valid if $0 < |z| < 1$

$$
\begin{aligned}
(1+z)^{-1} &= 1 - z + z^2 - z^3 + \dots \\(1-z)^{-1} &= 1 + z + z^2 + z^3 + \dots \\(1+z)^{-2} &= 1 - 2z + 3z^2 - 4z^3 + \dots \\(1-z)^{-2} &= 1 + 2z + 3z^2 + 4z^3 + \dots
\end{aligned}
\label{formula}
$$

## Singular Points

Take all $n$ points $(\pm n\pi, \pm 2n\pi, \dots)$

### Isolated Point

No other singular point in close neighborhood

### Poles

isolated points are poles too

poles of order $m=1$ are simple poles
