Deals with situations where there are conflict of interest between 2 opponents called ‘players’, who may have finite/infinite strategies or alternatives.

These are 2 person zero-sum games, because of the gain of one player is equal to loss to the other

Associated with each player is the payoff that one player pays to the other with respect to the each pair of strategies

The game is summarized in terms of the payoff to one player.

For 2 players with $m$ and $n$ strategies respectively, the payoff matrix will be $P_{m \times n}$

If $A$ and $B$ use strategy $i$ and $j$ respectively, then payoff to

- player A: $p_{ij}$
- player B: $-p_{ij}$

## Examples

- advertising campaigns for competing products
- planning war strategies for opposing armies

## Optimal Solution

Due to presence of conflict of interest, optimal solution selects the strategies for each player such that any change in strategies will not improve the payoff for either of the players.

## Simple Strategies

Parties can only pick 1 strategy each.

We are trying to find the best variation of the worst-case scenario for both parties

1. Strategy of $A$ is the strategy for which payoff = max(min) for A
   1. Find row-wise min
   2. Calculate the max of these
2. Strategy of $B$ is the strategy for which payoff = min(max) for B
   1. Find row-wise max
   2. Calculate the min of these
3. Saddle point solution = $(i, j)$, where $i$ and $j$ are the strategies that 
   - if $i=j$, neither $i, j$ would be willing to change their strategy
4. Value of game $= [\text{Sol}_A, \text{Sol}_B]$

## Mixed Strategies

$\not \exist$ Saddle point $\implies$ There is no single strategy for one/more players.

1. Consider strategies

   1. $A$ selects strategies $i \in [1, 2, \dots]$ w/ probability $x_i$, such that $\sum x_i = 1$
   2. $B$ selects strategies $i \in [1, 2, \dots]$ w/ probability $y_i$, such that $\sum y_i = 1$

2. Draw table of B’s picked strategy and A’s expected payoff

$$
\text{Expected Payoff}_A = 
$$

| B’s Strategy | A’s expected payoff |
| ------------ | ------------------- |
|              |                     |

3. Draw graph

4. idk

   - Maxmin = highest point of lower intersection open area in the graph
   - Minmax = lowest point of highest intersection open area in the graph

5. Equate the expected pay-off of the lines that are involved in maxmin

6. Value of game = value obtained by substituting $x_1$ in the intersecting equations (intersecting equations will give the same value)

7. Find the other person’s 

$$
\text{Expected Payoff}_B = 
$$

| A’s Strategy | B’s expected payoff |
| ------------ | ------------------- |
|              |                     |