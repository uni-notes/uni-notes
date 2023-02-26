## Eigen Values

are the values of $\lambda$ that satisfy $\eqref{values}$

$$
| A - \lambda I | = 0
\label{values}
$$

### Properties

1. Eigen values of upper/lower $\triangle$r matrix = diagonal elements
2. No of eigen values = order of A
3. Sum of eigen values = Sum of diagonal elements
4. Product of eigen values = $|A|$
5. If eigen values of $A = \lambda$, then

   | Matrix   | Eigen Value         |
   | -------- | ------------------- |
   | $A^{-1}$ | $\frac{1}{\lambda}$ |
   | $A^n$    | $\lambda^n$         |
   | $A^T$    | $\lambda$           |

## Eigen Vectors

are the values of $X$ that satisfies $\eqref{vectors}$

$$
(A - \lambda I) X = 0
\label{vectors}
$$

Eigen vector(s) of $A$ = eigen vector(s) of $A^{-1}, A^n, A^T$

### Working

| Scenario               | Method                               |
| ---------------------- | ------------------------------------ |
| Repeating eigen values | back substitution                    |
| else                   | Cramer’s rule for 2 independent rows |

