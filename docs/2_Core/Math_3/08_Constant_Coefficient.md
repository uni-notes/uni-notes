## 2nd Order Homogeneous DE with constant coefficients

$$
y'' + py' + qy = 0
$$

where $p, q$ are constants

Consider $y = e^{mx}$ as a possible solution, where $m$ = unknown constant. So our goal is to find $m$.

Then

$$
y' = m \cdot e^{mx} \\
y' = m^2 \cdot e^{mx} \\
\implies
(m^2 \cdot e^{mx}) + p(m \cdot e^{mx}) + qe^{mx} = 0 \\
e^{mx} ( m^2 + pm + q ) = 0 \\
$$

### Auxiliary equation

$$
e^{mx} \ne 0 \\
\implies
( m^2 + pm + q ) = 0
$$

Solve this to get the value(s) of unknown $m$

| Roots             |                       | General Solution $y$                 |
| ----------------- | --------------------- | ------------------------------------ |
| real and distinct | $m_1, m_2$            | $c_1 e^{m_1 x} + c_2 e^{m_2 x}$      |
| equal roots       | $m_1 = m_2 = m$       | $e^{mx} (c_1 + c_2 x )$              |
| Complex roots     | $m_1, m_2 = a \pm ib$ | $e^{ax} (c_1\cos bx + c_2 \sin bx )$ |

## Boundary Value Problems

Using given ‘initial conditions’, we need to find the values of $c_1$ and $c_2$
