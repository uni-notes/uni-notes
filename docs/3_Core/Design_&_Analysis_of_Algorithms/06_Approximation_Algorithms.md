Algorithm that generates a feasible solution with value close to the optimal solution.

To reduce an time complexity of solving an optimization problem

1. We remove the requirement that the algorithm must always generate an optimal solution
2. We use an ‘Probabilistically Good Algorithm’, that almost always generates optimal solution

## NP Problem

A problem is an NP (nondeterministic polynomial time) problem if it is solvable in polynomial time by a nondeterministic Turing machine

A P-problem whose solution time is bounded by a polynomial is always also NP

eg: 0/1 knapsack, traveling salesperson

## Terminology

| Symbol                              | Meaning                                                      |
| ----------------------------------- | ------------------------------------------------------------ |
| $p$                                 | NP problem                                                   |
| $I$                                 | Instance of $P$                                              |
| $\overset{\star} F (I)$             | Opitmal solution to $I$                                      |
| $\hat F(I)$                         | Approximate solution to $I$                                  |
| $A$                                 | Algorithm that generates feasible solution to every $I$ of $P$ |
| $\hat F(I) < \overset{\star} F (I)$ | Maximization problem                                         |
| $\hat F(I) > \overset{\star} F (I)$ | Minimization problem                                         |

## Types of Approximation Algorithms

| Type       | Meaning                                                      | Application                                                |
| ---------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Absolute   | $\vert \hat F(I) - \overset{\star} F (I) \vert \le k$, for some constant $k$<br />for every $I$ of $P$ | Planar Graph Coloring<br />Maximum Programs Stored Problem |
| $f(n)$     | $\frac{\vert \hat F(I) - \overset{\star} F (I)  \vert}{\overset{\star} F (I)} \le f(n)$, for $\overset{\star} F (I) > 0$<br />for every $I$ of $P$ |                                                            |
| $\epsilon$ | $f(n)$ approximation algo for which $f(n) \le \epsilon$, where $\epsilon$ is some constant |                                                            |

## Planar Graph Coloring

Coloring vertices of a graph, such that no two adjacent vertices have the same color

**Goal:** Minimize no of colors used

A graph is planar if it can be represented by a drawing in the plane such that no edges cross.

The maximum no of colors required for a planar graph is 4

```pseudocode
Algorithm Acolor(V, E)
{
  if V = null
  	return 0
  else if E = null
  	return 1
  else if (G is bipartite)
  	return 2
  else
  	return 4
}
```

**Time Complexity:** $O(|V|+|E|)$, which is the time taken to check if graph is bipartite

## Maximum Program Stored Problem

Consider

- $n$ programs
- two disks with storage capacity of $L$ each. 
- list $l$ where $l_i$ is the storage required to store program $i$

### Goal

Determine max no of programs that can be stored on the disks, without splitting a program over the disks.

### Solution

1. Sort $l$ in **ascending** order (to maximize count; in knapsack, we don’t maximize count, we maximize profit)
2. Keep placing elements

### Algorithm

```pseudocode
Algorithm Pstore(l, n, L) 
{ 
	// sort l in ascending order

	i = 1
	for j=1 to 2
	{
		sum = 0; // amount (part) of disk j already assigned 
    while (sum + l[i]) <= L
    { 
	    write ("store program", i, "on disk", j)
	    sum = sum + l[i]
    	i = i + 1
    	if i > n
      	return
    }
  }
}
```

### Time Complexity

$O(n) + \underbrace{O(n \log n)}_{\text{Sorting}} = O(n \log n)$

The most optimal algorithm will have exponential time.