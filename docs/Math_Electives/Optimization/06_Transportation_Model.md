Special case of LP, which deals with shipping of commodites from $m$ sources to $n$ nodes

The number of basic vars and independent constraints will be $(m+n-1)$

It will always be a **minimization problem**, since transporation model deals with the shipping cost of commodities

To solve, we've to
1. [Find initial BFS](#find-initial-bfs)
2. [Find optimal solution](#find-optimal-solution)



## Matrix

$m \times n$

For every cell $v_{ij}$, make sure that
$$
\begin{aligned}
v_{ij} &= \text{argmin}(D_i, S_j) \\
\sum v_i &= D_i \\
\sum v_j &= S_j
\end{aligned}
$$

## Balanced Transportation

$$
\sum \text{Supply} = \sum \text{Demand}
$$

## Find initial BFS

(Basic Feasible Soln)

Any basic feasible solution will have $(m+n-1)$

|                |         |
| -------------- | ------- |
| Basic vars     | $v > 0$ |
| Non-basic vars | $v=0$   |

- If the problem is balanced, no issues
- Else add an extra row/column to compensate
  - If no penalties, cell costs = 0
  - If $\exists$ Penalties, cell costs = penalties

Soln can be find using one of the following methods

| Method                           | Steps                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| North-West Corner                | Traverse from top-left to bottom-right                       |
| Least Cost                       |                                                              |
| Vogel’s Approximation<br />(VAM) | 1. Calculate row-wise penalties $P_i = \min_2(v_i) - \min(v_i)$<br />2. Calculate row-wise penalties $P_j = \min_2(v_j) - \min(v_j)$<br />3. Pick the row/column with max penalty. Cross out the penalty (and hence entire row/column), if the row/column is completely utilized<br />4. In that row/column, we select the cell with least cost<br />5. Repeat, excluding the recently-allocated cell |

## Find optimal solution

Method of multipliers. 2 conditions need to be satisfied

- Supply limits & demand requirements remain satisfied
- Shipments through all routes must be $\ge 0$

Checking optimality

1. Set $u_1 = 0$

2. Find $u_i, u_j$ for cells with $u_i + v_j = c_{ij}$
3. Find BIJ for **empty** cells: $\text{BIJ} = u_i + v_j-c_{ij}$
4. If BIJ $\le 0$ for **empty** cells, solution is optimal (this is minimization problem)
5. Else, non-optimal

Optimizing

1. Find the entering var as the empty cell with the most +ve BIJ
2. Put $\theta$ there
3. Make a square/rectangle, with corners as non-empty cells or $\theta$ cell
4. Add
   1. $-\theta$ to corner cells in the same row/col as $\theta$ cell
   2. $+\theta$ to other corner cells
5. $\theta$ = Max value that $\theta$ can assume, which is obtained using the cells in the same row/column of $\theta$ cell
6. Evaluate all the cells

If 0 appears in non-basic var, and there is no other potential entering var, it implies that optimality is already reached, and future iterations give Alternative Optima.

## Degenerate BFS

If in a cell we find zero mentioned, it means that all corresponds to basic var that has assumed $v=0$. It implies degenerate basic feasible solution.

## Maximization Problem

For eg, if we want to maximize the distribution of foods (not worried about money, for eg during a natural disaster).

The values of $c_{ij}$ will be the +ve output. Since transportation problem is always minimization.

1. $c_{ij} = -1 \times c_{ij}$
2. Solve as usual
3. Total output = -1 x total cost
