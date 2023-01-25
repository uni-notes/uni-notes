## Discrete Random Variables

takes finite/countably-infinite no of values

## PDF

Probability Density Function

$$
\begin{align}
f(x) &= P(X = x) \\f(x) &\ge 0 \\\sum f(x) &= 1
\end{align}
$$

## CDF

Cumulative Distribution Function

$$
\begin{align}
F(x) &= P(X \le x) \\&= \sum\limits_0^x f(x) \\ \\
P(a \le X \le b) &= \sum\limits_a^b f(x)
\end{align}
$$
## Terms

|                     |  Notation  |           Formula           |
| :-----------------: | :--------: | :-------------------------: |
|       $E(x)$        |   $\mu$    |     $\sum x \cdot f(x)$     |
|      $E(x^2)$       |            |    $\sum x^2 \cdot f(x)$    |
|       $V(x)$        | $\sigma^2$ |     $E(x^2) - [E(x)]^2$     |
|    $\rm{SD}(x)$     |  $\sigma$  |       $\sqrt {V(x)}$        |
| Normalised Variable |    $z$     | $\dfrac{x - E(x)}{\rm{SD}}$ |

$$
\begin{align}
E(k) &= k & E(kx) &= k \cdot E(x) & E(z) &= 0\\V(k) &= 0 & V(kx) &= k^2 \cdot V(x) & V(z) &= 1
\end{align}
$$

## Distributions

| Distribution   |                                                      |                            $f(x)$                            |            $\mu$             |                            $V(x)$                            |
| -------------- | ---------------------------------------------------- | :----------------------------------------------------------: | :--------------------------: | :----------------------------------------------------------: |
| Binomial       | - 2 outcomes<br />- independent and identical trials |              $nC_x \cdot p^x \cdot (1-p)^{n-x}$              |             $np$             |                          $np(1-p)$                           |
| Hypergeometric | dependent trials without replacement                 | $f(x) = \frac {MC_x \times (N-M) C_{(n-x)} } {NC_n} \\ \rm{max}\Big(0, n- (N-m) \Big) \le x \le \rm{min}(n, M)$ | $n \left(\dfrac M N \right)$ | $\left( \dfrac{N-n}{N-1} \right) \cdot n \cdot \dfrac M N \left( 1 - \dfrac M N \right)$ |
| Poisson        | discrete phenomenon in continuous interval           |    $\dfrac {e^{-\mu} \times \mu^x} {x!}, \mu = \alpha t$     |          $\alpha t$          |                          $\alpha t$                          |

### Rate Parameter $(\alpha)$

occurences per unit interval

$\alpha = \dfrac 1 \beta$

($\beta$ will be discussed in next topic)
