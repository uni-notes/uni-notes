## Continuous Random Variable

takes value on a continuum of scale, ie, can take **any** decimal value

## PDF

$$
\begin{align}
f(x) &\ge 0 \\
\int f(x) \ \mathrm{d} x &= 1
\end{align}
$$

## CDF

$$
\begin{align}
F(x) &= P(X \le x) \\&= \int\limits_{- \infty}^x f(x) \ \mathrm{d} x \\
P(a \le X \le b) &= P(a < x < b) \\&= \int\limits_a^b f(x) \ \mathrm{d} x
\end{align}
$$

## Terms

|          |            Formula            |
| :------: | :---------------------------: |
|  $E(x)$  |  $\int x \cdot f(x) \ \mathrm{d} x$  |
| $E(x^2)$ | $\int x^2 \cdot f(x) \ \mathrm{d} x$ |

(others are the same as discrete)

## Distributions

| Distribution                 |                                     |                                                              |            $\mu$            |              $V(x)$              |
| ---------------------------- | :---------------------------------: | :----------------------------------------------------------: | :-------------------------: | :------------------------------: |
| Uniform                      |                                     | $f(x) = \begin{cases} \frac 1 {B-A} & A \le x \le B \\ 0 & \text{elsewhere} \end{cases}$ |      $\dfrac {B+A} 2$       |     $\dfrac 1 {12} (B-A)^2$      |
| Normal                       |          Symmetric about 0          | $f(x) = \dfrac {1}{\sigma \sqrt{2\pi}} \exp \left\{ \dfrac {-1}{2} \left(\dfrac{x-\mu}{\sigma} \right)^2 \right\} \\
\begin{align} P(x<k) &= P \left(z<\frac{k-\mu}{\rm{SD}} \right) \\ P(x>k) &= P(x < -k) \end{align}$ |              0              |                1                 |
| t                            |              Symmetric              |                                                              |              0              |                >1                |
| Binomial $\to$ Normal Approx |   $np \ge 10$ or $n(1-p) \ge 10$    | $\begin{align} x' &= x \pm 0.5 \\ z &= \frac {x' - \mu}{\rm{SD}} \end{align}$ |            $np$             |            $np(1-p)$             |
| Gamma                        |     time between $n$ occurences     | $f(x) = \dfrac{1}{B^\alpha \lceil\alpha} \cdot x^{\alpha-1} \cdot e^{-\frac x \beta}$ |       $\alpha \beta$        |         $\alpha \beta^2$         |
| Exponential                  | time between successive/consecutive |            $f(x) = \lambda \cdot e^{-\lambda x}$             | $\dfrac 1 \lambda  = \beta$ | $\dfrac 1 {\lambda^2} = \beta^2$ |

- $\lambda$ = mean no of occurances per unit time
  $\lambda = \alpha\text{(poisson)}$
- $\beta$ = mean time b/w occurances
  $\beta = \frac 1 \lambda = \frac 1 {\alpha\text{(poisson)}}$
- $\alpha$ = shape parameter
  it is the average number of occurences of an event
