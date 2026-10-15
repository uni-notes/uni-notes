# Goal Programming

This deals with situations with multiple objective functions.

Sometimes, some goals will be more important than others

## Deviation Variables

Represent the amount by which goal will be violated

|         | Deviation ___ RHS of constraint |
| ------- | ------------------------------- |
| $s_i^+$ | above                           |
| $s_i^-$ | below                           |

$s_i^+$ and $s_i^-$ are by definition dependent and hence cannot be taken as basic variables simultaneously. This means that in any simplex iteration, $\le 1$ one the 2 deviation variables can assume +ve values

