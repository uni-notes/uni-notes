## Set

collection of well-defined elements

## Vector Space

A non-empty set $V$ with binary operations $\oplus$ and $\odot$, which satisfies the following rules

| Law                            |                                                              |
| ------------------------------ | ------------------------------------------------------------ |
| Closure Law wrt Addition       | $\forall u,v \in V, \quad \vec u \oplus \vec v \in V$        |
| Commutative Law wrt Addition   | $\vec u \oplus v = \vec v \oplus u$                          |
| Associative Law wrt Addition   | $(\vec u \oplus \vec v) \oplus \vec w = \vec u \oplus (\vec v \oplus \vec w)$ |
| Existence of additive identity | For $\vec u \in V$, there exists $\vec 0 \in V$ such that<br />$\vec 0 \oplus \vec u = \vec u \oplus \vec 0 = \vec u$<br />$\vec 0$ is not necessarily $(0, 0)$ |
| Existence of additive inverse  | For $\vec u \in V$, there exists $-u \in V$ such that<br />$\vec u \oplus (- \vec u) =   (-\vec u) \oplus \vec u = \vec 0$ |
| Closure Law wrt multiplication | For any scalar $\alpha$ (any real no) and $\vec u \in V$<br />$\alpha \odot \vec u \in V$ |
| Distributive Law (Right-Side)  | For $u, v \in V$ and scalar $\alpha$<br />$\alpha \odot (\vec u \oplus \vec v) = (\alpha \odot \vec u) \oplus (\alpha \odot \vec v)$ |
| Distributive Law (Left-Side)   | For $u \in V$ and scalars $\alpha, \beta$<br />$(\alpha + \beta) \odot \vec u = (\alpha \odot \vec u ) \oplus (\beta \odot \vec u)$ |
| Distributive Law (Variation)   | For $u \in V$ and scalars $\alpha, \beta$<br />$(\alpha \beta) \odot \vec u = \alpha \odot (\beta \odot \vec u)$ |
| Existence of unity             | For $u \in V$<br />$1 \odot \vec u = \vec u$                 |

## Known Vector Spaces

- Real numbers
- $R_2, R_3, R_n$
- matrices
- polynomials
    - form $ax^n + bx^{n-1} + \dots + \alpha, \quad a, b \in R, \quad n \in Z$
    - $P_n$ means degree of the polynomial $\le n$
- continuous functions

## Subspace

Let $S \subset V$ vector space. Then, $S$ is a subspace if

1. $\vec 0 \in S$
2. $\forall u, v \in S, \quad u \oplus v \in S$
3. $\forall u \in S, \quad \alpha \odot u \in S$

Trick to identify is if sum of powers of multiplicative terms is 1
For eg, $x^a y^b + w^c z^d$ is subspace if $a + b= 1, c + d = 1$

## Polynomial

$P_n$ is a polynomial where degree $\le n$

For eg, even $(1 + x)$ is $P_3$, as degree $= 1 \le 3$