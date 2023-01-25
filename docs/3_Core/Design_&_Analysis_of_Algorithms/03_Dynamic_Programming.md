Difference b/w divide & conquer and dynamic programming is 
- divide & conquer combines the solutions of the subproblems to find solution of main problem
- dynamic programming uses the result of the subproblems to find optimum solution of main problem

Difference b/w greedy method and dynamic programming is

- greedy method first makes a choise that appears best at the time, and then solves a resulting subproblem
- dynamic programming solves all subproblems, and then selects one that helps to find the optimum solution

## Principle of Optimality

Optimal sequence of decisions has the property that whatever the initial state and decision are, the remaining decisions must constitute an optimal decision sequence with regard to the state resulting from the first decision.

## All Pair Shortest Path

$$
\begin{aligned}
A^k(i, j) = \text{argmin} & \{ \\
& A^{k-1} (i, j), \\
& A^{k-1} (i, k) + A^{k-1} (k, j) \\
&\}
\end{aligned}
$$

Try to find the path between pairs of points either directly/through another intermediate point.

Time complexity $= O(n^3)$

```pseudocode
Algorithm AllPaths(cost, n)
{
	for k=1 to n // taking every node as intermediary
		for i=1 to n
			for j=1 to n
				a[i][j] = argmin(
					cost[i][j],
					cost[i][k] + cost[k][j]
				)
				
	return a
}
```

## Single Source Shortest Path

Also called as Bellman Ford algorithm

Similar to [All Pair Shortest Path](#All-Pair-Shortest-Path), but only from a single source to every other point.

```pseudocode
Algorithm BellmanFord(v, cost)
{
	for i=1 to n
		dist[i] = cost[v][i]

	for k=1 to n
		for i=1 to n
			for j=1 to n
				a[i] = argmin(
					a[i],
					cost[i][k] + cost[k][j]
				)
		
	return a
}
```

Time complexity $= O(n^3)$
