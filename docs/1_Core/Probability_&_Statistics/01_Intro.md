# Introduction

## Goals

1. Summary statistics: Describe/summarize a large set of data with a few ‘statistics’
2. Statistical inference: Use sample data to infer population characteristics

## Probability vs Statistics

- Probability: Predict behavior of sample given known knowledge of population
- Statistics: Infer properties of population given knowledge of sample

The two are tied together by sampling distribution

## Approaches

|                  | Frequentist                                                  | Bayesian                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Probability      | Limiting case of repeated measurements                       | Subjective, based degree of certainty in the event           |
| Data             | Random variable                                              | Constant                                                     |
| Model parameters | Unknown constant                                             | Unknown random variable                                      |
| Basis            | Weak law of large numbers<br />Assumes IID                   |                                                              |
| Limitations      | Not optimal for rare events                                  |                                                              |
| Intervals        | Confidence Intervals<br /><br />With large number of repeated samples, $\alpha \%$ of such calculated confidence intervals would include the true value of the parameter | Credible Intervals<br /><br />Estimated parameter has a $95 \%$ probability of falling within the given interval |
| Statistics       |                                                              | Use prior belief to systematically update knowledge after experiment, through Bayes theorem |

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

