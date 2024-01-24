1. Divide problem into 2/more smaller sub-problems (**Divide**)
2. Solve sub-problems recursively
3. Obtain sol to original problem by combining sub-solutions (**Conquer**)

## General Form

```pseudocode
Algorithm divide_and_conquer(a, p, r)
{
	if(p<r)
	{
		part1 = divide_and_conquer(a, p, q_1)
		part2 = divide_and_conquer(a, q1 + 1, q2)
		...
		partn = divide_and_conquer(a, qn + 1, r)
		
		return combine(part1, part2, ..., partn)
	}
	else
	{
		return solution
	}
}
```

## Algorithms

|                   | $T(n)$                                 | $O()$                                  |
| ----------------- | -------------------------------------- | -------------------------------------- |
| Binary Search     | $1 \cdot T(n/2) + 1$                   | $O(\log n)$                            |
| Min-Max Algorithm | $2 \cdot T(n/2) + 1$                   | $O(n)$                                 |
| Merge Sort        | $2 \cdot T(n/2) + n$                   | $O(n \log n)$                          |
| Quick Select      |                                        |                                        |
| Quick Sort        | $2 \cdot T(n/2) + n$<br />$T(n-1) + n$ | $O(n \log n)$<br />Worst-Case $O(n^2)$ |
| $n!$              | $T(n-1) + 1$                           |                                        |
| Tower of Hanoi    | $2 T(n-1) + 1$                         |                                        |
| Example           | $3T(n/4) + n^2$                        |                                        |

Derive the complexity for the above using

- Substitution method
- Recursion tree
- Master’s Theorem

### Min-Max

```pseudocode
Algorithm min(a, p, r)
{
  if (p<r)
  {
    q = (p+r)/2;
    
    min1 = min(a, p, q);
    min2 = min(a, q+1, r);
    
    return argin(min1, min2);
  }
  else
  {
	  return a[p];
  }
}

// We could use brute-force method. It is O(n) as well, but it worse.

Algorithm min_max(a)
{
  min = max = a[0];
  
  for i=1 to n
  	if a[i] < min
  		min = a[i]
  	else if a[i] > max
  		max = a[i]
  
  return min
}
```

### Quick Select

The algorithm is quick select algorithm, based on the Quick-Sort algorithm.

It returns $k^\text{th}$ smallest element in S


```pseudocode
Select (k, S)
{
  pick pivot in S
  
  partition S into L, E, Q such that:
    max(L) < pivot // L contains all elements smaller than pivot
    E = {pivot}
    pivot < min(G) // G contains all elements greater than pivot
  
  if k ≤ length(L) // Searching for item ≤ pivot.
    return Select(k, L)
  else if k ≤ length(L) + length(E) // Found
    return pivot
  else  // Searching for item ≥ pivot.
    return Select(k - length(L) - length(E), G)
}
```

#### Time Complexity

- Worst-Case $O(n^2)$
- Average-Case $O(n)$


### Others

Refer DSA

## Master’s Theorem

Consider a recurrence relation

$$
T(n) = a T \left( \frac{n}{b} \right) + n^d
$$

| $d \ \_\_\_ \ \log_b a$ |     $T(n)$     |
| :---------------------: | :------------: |
|           $>$           |     $n^d$      |
|           $=$           |  $n^d \log n$  |
|           $<$           | $n^{\log_b a}$ |
