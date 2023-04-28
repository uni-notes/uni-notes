## Formulae

$$
\begin{aligned}
P(S) &=1 \\
0 \le P(A) &\le 1 \\
P(A') &= 1 - P(A) \\
P(A \cup B) &= P(A) + P(B) - P(A \cap B) \\
P(A \cup B \cup C) &= P(A) + P(B) + P(C) - P(A \cap B) - P(B \cap C) - P(A \cap C) + P(A \cap B \cap C) \\
P(A \cap B') &= P(A) - P(A \cap B) \\
&= P(A \cup B) - P(B)
\end{aligned}
$$

## Cases

| Case                | Property                        |
| ------------------- | ------------------------------- |
| Mutually-Exclusive  | $P(A \cap B) = 0$               |
| Mutually-Exhaustive | $P(A \cup B) = 1$               |
| Independent         | $P(A \cap B) = P(A) \cdot P(B)$ |

2 events are independent if one event does not affect the occurance of the other

## No of ways

|                           | When to use                                                  | No of ways of selection                                   |
| ------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Product Rule              | there are $k$ elements, and each have different ways of selection | $n_1 \times n_2 \times \dots \times n_k$                  |
| Permutation               | some sort of ordering                                        | $nP_r = \frac{n!}{(n-r)!}$                                |
| Combination               |                                                              | $nP_r = \frac{n!}{r!(n-r)!}$                              |
| Indistinguishable Objects | there are $k$ objects, such that $x_1 + x_2 + \dots + x_k = n$, where $x_1, x_2, \dots$ are the no of elements of that type | $\frac{n!}{x_1 ! \times x_2 ! \times \dots \times x_k !}$ |

## Conditional Probability

**Probability of A given B** is the probability of A occuring given that A has already occured

$$
P(A|B) = \frac{P(A \cap B)}{P(B)} \quad P(B) \ne 0
$$
