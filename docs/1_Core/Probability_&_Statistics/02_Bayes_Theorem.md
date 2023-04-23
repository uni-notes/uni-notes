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

### General Formula

$$
\begin{aligned}
P(A_1|B)
&= \frac{P(A_1 \cap B)}{P(B)} \\
&= \frac{P(B | A_1) \cdot P(A_1)}{\sum\limits_{i=1}^{n} P(B|A_i) \cdot P(A_i)} \\
\end{aligned}
$$

where $A_1, A_2, A_3, \dots, A_n$ are all mutually exclusive events

### Phrases

- “out of”
- “of those who”

### Given

- $P(A_1)$
- $P(A_2)$
- $P(B|A_1)$
- $P(B|A_2)$