## Common Definitions
- $y$ is the target: No of failures
- $x$ is the set of relevant features such as type of transformer, area, temperature

## Goal

Obtain a model $\hat f$ which gives estimates $\hat y$ from $x$

$$
\begin{aligned}
\hat y
&= \hat f(x)
\end{aligned}
$$

## Poisson is not the most ideal

$$
\begin{aligned}
\hat y
&= \text{Poisson}(x)
\end{aligned}
$$

- Poisson assumes that variance increases with mean
- Poisson is just an approximation of Binomial distribution when
	- $n$ is large: many transformers
	- $p$ is small: low failure rate
- This approximation is used
	- when $n$ is unknown, ie no of transformers is unknown
	- for numerical performance, but not a concern nowadays

## Better to

$$
\begin{aligned}
\hat y
&= \hat f(x) \\
&= n \times \hat p(\text{Failure} \vert x_i)
\end{aligned}
$$

- Predict the probability of failure of each type of transformer using 'logistic/binomial' model
- Count of failure = $n p$
	- $n=$ number of transformers
	- $p =$ probability of failure of each transformer

Advantages
- Probability of failure for each $x$ is insightful
- Assumption-free
	- Better uncertainty quantification