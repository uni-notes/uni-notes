The undetermined coefficient method is possible for a few standards functions such as ==$R(x) = e^{ax}, \sin(ax), \cos(ax),$ polynomials==. This method also requires a trial solution to compute the required **particular solution**.

**Note**
If RHS $= \sin(2x) + \cos(2x)$, then we can consider it as a single function $R(x)$

Consider

- Constants $l, k, a, b \in R$
- **Undetermined Coefficients** $A, B, A_0, A_1, \dots, A_n$ (unknown constant)

The exception cases are for preventing duplication of terms, and hence prevent linear dependency of the solutions.

| $R(x)$                                                       | Trial Particular Solution           | Exception based on root $m$ of auxilary eqn |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------------- |
| $l e^{ax}$                                                   | $A e^{ax}$                          |                                             |
|                                                              | $A x e^{ax}$                        | $m_1 = a$ or $m_2 = a$                      |
|                                                              | $A x^2 e^{ax}$                      | $m_1 = m_2 = a$                             |
| $l \cos(ax)$<br />$l \sin(ax)$<br />$l \cos(ax) \pm k \sin(ax)$ | $A \cos ax + B \sin ax$             |                                             |
|                                                              | $x (A \cos ax + B \sin ax)$         | $m= 0 \pm ai$                               |
| $a_0 + a_1 x + \dots + a_n x^n$<br />($n^{\text{th}}$ degree polynomial) | $(A_0 + A_1 x + \dots + A_n x^n)$   |                                             |
|                                                              | $x(A_0 + A_1 x + \dots + A_n x^n)$  | $m = 0$                                     |
| $e^{ax} \cos bx$<br />$e^{ax} \sin bx$<br />$e^{ax} ( \cos bx + \sin bx )$ | $e^{ax} ( A \cos bx + B \sin bx )$  |                                             |
|                                                              | $xe^{ax} ( A \cos bx + B \sin bx )$ | $m = a \pm bi$                              |

## Trick for product of 3 functions

If $y_g$ and the trial particular solution are similar

- instead of using $(uvw)' = uvw' + uv'w + u'vw$
- we can take

$$
x e^x ( A \cos x + B \sin x) \to x \phi
$$

### Example

$$
\begin{aligned}
y'' - 2y' + 2y &= e^x \sin x \\
x e^x ( A \cos x + B \sin x) & \to x \phi \\
{y_g}'' - 2{y_g}' + 2{y_g} &= 0 \\
\implies \phi'' - 2\phi' + 2\phi &= 0
\end{aligned}
$$

