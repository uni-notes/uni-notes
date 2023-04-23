## Collision Handling

eliminates collisions in hashing

|                 | Collision                                      | Search Complexity | Disadvantage |
|---               | ---                                             | ---        | ---|
|Separate Chaining | array with $N$ buckets pointing to linked lists | $O(n)$     | memory wastage|
|Linear Probing    | $(i + j )\% N$                                 | $O(1)$     | clustering|
|Quadratic Probing | $(i + j^2) \% N$                               |            | some elements may not be able to stored|
|Double Hashing    |                                                 |            ||

Problem with probing is the possibility of full bucket

## Linear Probing Search

1. Compute $i = h(k)$

2. Start at array cell $a[i]$

3. Probe consecutive locations until

   | return      | case       |
   | ----------- | ---------- |
   | present     |            |
   | not present | empty cell |

   something more

## Double Hashing

use a secondary hash function $d(k)$

### Hashing

$$
\begin{aligned}
h(k) &= k \% N \\
d(k) &= q - k \% q \\
\end{aligned}
$$

where

1. $q$ is prime and $q < N$
1. $N$ is the no of elements

### Bucket Placement

$$
\begin{aligned}
\text{index} = \Big( i+ jd(k) \Big)
\% N \\
i &= h(k) \\
j &= 0, 1,\dots
\end{aligned}
$$

## Load Factor

$$
\lambda = \frac{n}{N}
$$

- $n$ is the no of keys
- $N$ is the no of buckets

Load Factor should preferably be $\lambda < 0.75$, or atleast $\lambda < 1$

## Rehashing