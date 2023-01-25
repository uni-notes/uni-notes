## General Form of 2nd order DE

$$
F(x, y, y', y'') = 0

$$

This is for variable coefficients.

## Solving

- 2nd order DE is reduced into two 1st order DE
- they are solved one after each other

Reduction of order method is possible under 2 cases

|               | Case 1                     | Case 2                                                       |
| ------------- | -------------------------- | ------------------------------------------------------------ |
| missing terms | Dependent variable $y$     | Independent variable $x$                                     |
| Form          | $F(x, y', y'') = 0$        | $F(y, y', y'') = 0$                                          |
| Let           | $y' = P \implies y'' = P'$ | $y' = P \\
\implies y'' = P' = \frac{dP}{dy} y' \\ y''= P \left(\frac{dP}{dy}\right)$ |
| Solve         | $F(x, P, P') = 0$          | $F(y, P, P \frac{dP}{dy}) = 0$                               |
| Substitute    | $y' = P \implies y'' = P'$ | $y' = P \implies y'' = P'$                                   |
| Solve         | $F(x, y)$                  | $F(x, y)$                                                    |
