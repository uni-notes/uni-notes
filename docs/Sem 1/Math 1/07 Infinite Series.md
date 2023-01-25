## Infinite Series

Let $\set{a_n}_{n \in \mathbb{Z^+}}$ be a sequence. Then $\sum\limits_{n = 1}^\infty a_n = a_1 + a_2 + \dots$ is called a series.

If the series has a finite number of terms, it is called a finite series; otherwise it is called an infinite series.

A finite series is always convergent.

A infinite series may/ may not be convergent

|                            Series                            |                             Type                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| $1 + \frac{x}{1!} + \frac{x^2}{2!} + \dots + \frac{x^n}{n!} = e^x$ |                      Converges to $e^x$                      |
|                       $1 + 1 + \dots$                        |                          Divergent                           |
|               $1 + \frac12 + \dots + \frac1n$                |                          Divergent                           |
|                   $1 - 1 + 1 - 1 +  \dots$                   | Neither convergent/divergent<br />it is an alternating series which oscilates |

If we are able to find the sum of a series, then the series converges to the sum $S_n = \dfrac{a}{1 - r}$

- if sum is finite, then convergent series
- else, divergent series

## Series of +ve Terms

Consider series $\sum\limits_{n = 1}^\infty a_n = a_1 + a_2 + \dots + a_n$. This series is a series of +ve terms as $a_n \ge 0, \forall n$.

We use the following tests.

## $n^\rm{th}$ Term Test

$$
\lim\limits_{n \to \infty} a_n = 
\begin{cases}
\ne 0 & \text{Divergent} \\= 0 & \text{Test fails}
\end{cases}
$$

### Important Results

- Geometric sum $a + ar + ar^2 + \dots$
  - converges to $\dfrac{a}{1 - r}, |r| < 1$
  - diverges

|                  |                                             |                    Converges                     |    Diverges     |
| :--------------: | :-----------------------------------------: | :----------------------------------------------: | :-------------: |
| Geometric Series |           $a + ar + ar^2 + \dots$           | $|r| < 1$<br />converges to $\dfrac{a}{1-r}$ | $|r| \ge 1$ |
|     p-series     | $\sum\limits_{n = 1}^\infty \dfrac{1}{n^p}$ |                     $p > 1$                      |    $p \le 1$    |

$$
\begin{align}
\lim\limits_{n \to \infty} \frac { \ln |n| }{n} &= 0 \\(\ln |n| \text{ always} &< n, \text{ so den reaches } \infty \text{ faster} ) \\ \\
\lim\limits_{n \to \infty} x^{\frac 1n} &= 1 \\\lim\limits_{n \to \infty} n^{\frac 1n} &= 1 \\(x^0 = n^0 &= 1) \\ \\
\lim\limits_{n \to \infty} \left( 1 + \frac x n \right)^n &= e^x \\\lim\limits_{n \to \infty} x^n &= 0 \text{ if } |x| < 1 \\\lim\limits_{n \to \infty} \frac{x^n}{n!} &= 0 \\(n! &> x^n),  \text{ when $n$ is large so den reaches $\infty$ faster}

\end{align}
$$

## Integral Test

This test can be applied when $a_n = f(n)$ is integrable

Let

- $\sum a_n$ be a series of +ve terms
- $a_n = f(n)$ where $f$ is
  - continuous
  - +ve
  - decreasing function of $n$, for some $n \ge N$

Then by integral test, $\int\limits_N^\infty f(x) \ dx$ and $\sum\limits_N^\infty a_n$ converge/diverge together

|   $I$    |                                                   |
| :------: | :-----------------------------------------------: |
|  Finite  | Converges<br />(basically $S_n$ is finite number) |
| Infinite |                     Diverges                      |

## Ratio Test

Used when series contains factorials like $n!, (2n)!$

Let $\sum a_n$ be a series of +ve terms.

Let $\lim\limits_{n \to \infty} \dfrac{a_{n+1}}{a_n} = k$

|  $k$   |            |
| :----: | :--------: |
| $< 1$  | Converges  |
| $> 1$  |  Diverges  |
| $0, 1$ | Test Fails |

## Root Test

Used when series contains terms with exponents, such as $n^n, n^{n+1}, n^\frac1n$

Let $\lim\limits_{n \to \infty} (a_n)^\frac1n = k$

|  $k$  |            |
| :---: | :--------: |
| $< 1$ | Converges  |
| $> 1$ |  Diverges  |
|  $1$  | Test Fails |

## Limit Comparison Test

Best used when $a_n$ is a fraction of polynomial, ie $a_n = \frac {P(n)}{Q(n)}$, where $P, Q$ are polynomials in terms of $n$

Let

- $\sum a_n$ be a series of +ve terms

- $\sum b_n$ be a known series (we know if it converges/diverges)

  - We choose $b_n = \dfrac{1}{n^{q-p}}$, where

    - P = degree of numerator
    - Q = degree of denominator

  - If $b_n$ is a p-series of the form $\sum \dfrac{1}{n^p}$
    |   $p$   |           |
    | :-----: | :-------: |
    |  $> 1$  | converges |
    | $\le 1$ | diverges  |
    
  
- $\lim\limits_{n \to \infty} \frac {a_n}{b_n} = k$

Then

|               Given               |                                         |
| :-------------------------------: | :-------------------------------------: |
|          $k = c (\ne 0)$          | both $\sum a_n$ and $\sum b_n$ converge |
|    $k = 0, \sum b_n$ converges    |          $\sum a_n$ converges           |
| $k \to \infty, \sum b_n$ diverges |           $\sum a_n$ diverges           |
