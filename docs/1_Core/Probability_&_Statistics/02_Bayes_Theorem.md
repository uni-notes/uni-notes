## Bayes’ Theorem

It determines the probability of an event with uncertain knowledge.  

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

where
- $P(A|B)$ = posterior,  
- $P(B|A)$ = likelihood,  
- $P(A)$ = prior probability
- $P(B)$ = marginal probability 

## General Formula

$$
\begin{aligned}
P(A_i|B)
&= \frac{P(A_i \land B)}{P(B)} \\
&= \frac{P(B | A_i) \cdot P(A_i)}{\sum\limits_{j=1}^{n} P(B|A_j) \cdot P(A_j)} \\
\end{aligned}
$$

where $A_1, A_2, \dots, A_n$ are all mutually exclusive events

## Conditions

1. Events must be disjoint (no overlapping)
2. Events must be exhaustive: they combine to include all possibilities

## Phrases

- “out of”
- “of those who”

### 