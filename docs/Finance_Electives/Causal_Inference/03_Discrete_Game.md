# Discrete Game

Individual choices involve strategic interactions, which we can model
using game theory.

Similar to discrete choice models, but here individuals’ choices affect each other’s choices

Each individual’s choice can be considered the result of an equilibrium strategy. The resulting model is called a discrete game model.
$$
u_{ij} = f(x_{ij}) + g(x_{i' j}) + \epsilon_{ij} \\
i = \text{other individuals}
$$
Assume that players simultaneously make their decisions without knowing
what the other players will do. Such a game is called a game of incomplete information.
Because players do not know what others will do, their choices will depend on their beliefs
(expectations) about other players’ choice probabilities
$$
\begin{aligned}
u_{i0} &= \epsilon_i^0 \\
u_{i1} &= \pi_i(y_j) + \epsilon_i^1 \\
\\
y_i
&= \arg \max \{ u_{i0}, E_i[u_{i1}] \} \\
&= \arg \max \{ \\
& \quad  \epsilon_i^0, \\
& \quad  \pi_{i0} p_i (a_j=0) + \pi_{i1} p_i (a_j = 1) + \epsilon_i^1 \\
& \quad  \}
\end{aligned}
$$
where

- $\pi_i =$ profit function of $i$
- $y \in \{ 0, 1 \}$ is the choice
- $j$ is another player
- $p_i(y_j)$ is $i$’s belief of $j$’s probability of choosing $y_j$
- $E_i(y_j)$ is $i$’s belief of $j$’s expected utility from choosing $y_j$

Assumption:

- $\pi_{i0} > \pi_{i1}$ if players are competitors
- $\pi_{i0} < \pi_{i1}$ if players are complementers/symbiotic

## Nash Bayesian Equilibrium

In equilibrium, each player has the correct belief about the
choice probabilities of other players, ie $p_i(y_j) = p(y_j) , \forall i, j$

Assuming

- $\epsilon_i^0, \epsilon_i^1 \sim \text{Gumbel}(0, 1)$ 
- 

$$
\begin{aligned}
&p(y_i = 1) \\
&= \dfrac{
\exp[
\pi_{i0} \cdot p(y_j = 0) + \pi_{i1} \cdot p(y_j = 1)
]
}{
1 + \exp[
\pi_{i0} \cdot p(y_j = 0) + \pi_{i1} \cdot p(y_j = 1)
]
} \\
& \forall i, j
\end{aligned}
$$

