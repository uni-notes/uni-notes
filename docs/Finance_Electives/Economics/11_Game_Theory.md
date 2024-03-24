# Game Theory

## Auction

Consider $n$ risk-neutral bidders, with independent private value $v_i ~ F$

Each bidder knows their own $v_i$ and distribution $F$, but not the $v_i$ of others

Observed bids are the Bayesian Nash equilibrium outcome of the game
$$
\begin{aligned}
b_i
&= v_i - \dfrac{1}{F(v_i)^{n-1}} \int \limits_0^{v_i} F(x)^{n-1} \cdot dx \\
&= v_i - \dfrac{1}{n-1} \dfrac{G_n(b_i)}{g_n(b_i)}
\end{aligned}
$$
where

- $b_i$ is the bid amount
- $g_n$ is pdf of bid distribution
- $G_n$ is cdf of bid distribution

## Structural Estimation

1. For each auction, non-parametrically estimate $g_n$ and $G_n$ from observed bids $b_1, \dots, b_n$
2. For each bidder, calculate

$$
\hat v_i = b_i + \dfrac{1}{n-1} \dfrac{\hat G_n(b_i)}{\hat g_n(b_i)}
$$

3. Estimate $\hat F$ using $\hat v_i$
4. Predict winning bid

$$
E[\max \{ b_i \}] = E \Bigg[
\max \Big\{
v_i - \dfrac{1}{\hat F(v_i)^{n-1}}
\int\limits_0^{v_i} \hat F(x^{n-1}) \cdot dx
\Big\}
\Bigg]
$$

## Monopoly

