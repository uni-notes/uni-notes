## Discrete Random Variables

takes finite/countably-infinite no of values

## PDF

$$
\begin{aligned}
f(x) &= P(X = x) \\
f(x) &\ge 0 \\
\sum f(x) &= 1
\end{aligned}
$$

## CDF

$$
\begin{aligned}
F(x) &= P(X \le x) \\
&= \sum\limits_0^x f(x) \\
P(a \le X \le b) &= \sum\limits_a^b f(x)
\end{aligned}
$$

## Terms

|                     |  Notation  |           Formula           |
| :-----------------: | :--------: | :-------------------------: |
|       $E(x)$        |   $\mu$    |     $\sum x \cdot f(x)$     |
|      $E(x^2)$       |            |    $\sum x^2 \cdot f(x)$    |
|       $V(x)$        | $\sigma^2$ |     $E(x^2) - [E(x)]^2$     |
|    $\text{SD}(x)$     |  $\sigma$  |       $\sqrt {V(x)}$        |
| Normalised Variable |    $z$     | $\dfrac{x - E(x)}{\text{SD}}$ |

$$
\begin{aligned}
E(k) &= k & E(kx) &= k \cdot E(x) & E(z) &= 0\\
V(k) &= 0 & V(kx) &= k^2 \cdot V(x) & V(z) &= 1
\end{aligned}
$$

## Distributions

| Distribution      | Comment                                                                                                                            | Application           | Assumptions                    |                                                              $f(x)$                                                               |            $\mu$             |                                        $\sigma^2$                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------: | :--------------------------: | :--------------------------------------------------------------------------------------: |
| Bernoulli         | - 2 outcomes                                                                                                                       |                       | Independent & identical trial  |                                                                                                                                   |             $p$              |                                         $p(1-p)$                                         |
| Binomial          | $n$ indepedent Bernoulli events w/ replacement                                                                                     | Coin flips            | Independent & identical trials |                                                $nC_x \cdot p^x \cdot (1-p)^{n-x}$                                                 |             $np$             |                                        $np(1-p)$                                         |
| Hypergeometric    | $n$ dependent Bernoulli trials without replacement                                                                                 |                       |                                |      $f(x) = \frac{MC_x \times (N-M) C_{(n-x)} }{NC_n}$ <br /> $\text{max}\Big(0, n- (N-m) \Big) \le x \le \text{min}(n, M)$      | $n \left(\dfrac M N \right)$ | $\left( \dfrac{N-n}{N-1} \right) \cdot n \cdot \dfrac M N \left( 1 - \dfrac M N \right)$ |
| Geometric         | Negative binomial dist with $r=1$<br />No of failures before first success                                                         |                       |                                |                                                                                                                                   |                              |                                                                                          |
| Negative Binomial | 'Rare' discrete phenomenon in continuous interval                                                                                  | Counts, Rates<br>Risk | Follows Poisson Process        | $f_x(x) = \begin{cases} \begin{pmatrix} x-1\\ r-1 \end{pmatrix} p^r q^{x-r}, & x= r, r+1, \dots  \\ 0, & \text{o.w.} \end{cases}$ |       $\dfrac{rq}{p}$        |                    $\dfrac{rq}{p^2}$<br><br>$\mu + \dfrac{\mu^2}{k}$                     |
| Poisson           | Negative Binomial with:<br>variance = mean = constant<br><br>Poisson approximates Binomial when<br>- $p \to 0$<br>- $n \to \infty$ | Counts, Rates<br>Risk | Follows Poisson Process        |                                               $\dfrac {e^{-\mu} \times \mu^x}{x!}$                                                |     $\lambda = \alpha t$     |                                          $\mu$                                           |

### Rate Parameter $(\alpha)$

Occurences per unit interval

$\alpha = \dfrac 1 \beta$

($\beta$ will be discussed in next topic)

## Poisson Process

- Average time between events $\lambda$ is constant
- Average time between events $\lambda$ is known
- Rate of occurrence $\alpha$ is constant
	- Probability of $k$ arrivals in an interval is constant
- Occurence of events are independent of each other
- Memory-less process
	- $f(x \vert T \le t + \Delta t)$ is independent of $f(x \vert T \le t)$
- Events cannot occur simultaneously; there is at least a tiny interval between events