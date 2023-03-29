## Summary

$$
\begin{aligned}
r &= \left\lceil \frac n k \right \rceil \\n &= k(r-1) + 1 \\k &= \left\lceil \frac n r \right \rceil \\
\end{aligned}
$$

## Pigeon Hole Principle

if $m$ holes are assigned for $n$ pigeons, and $m<n$, then atleast one hole will have atleast 2 pigeons

in other words, if $k+1$ or more objects are places into $k$ boxes, then there is atleast one box containing two or more objects.

if $n$ objects are placed in $k$ boxes, then there is **atleast** one box with **atleast** $\lceil \frac n k \rceil$ objects

- ceiling $\lceil x \rceil$ means that $x$ is rounded up
- floor $\lfloor x \rfloor$ means that $x$ is rounded down

### Reason

A function from a finite set to a *smaller* finite set cannot be one-one, and hence there will be 2 elements in the domain that have the same image in the co-domain.

### Application

Minimum no of objects $n$ to be distributed among $k$ boxes such that $r$ objects must be in one of the boxes is given by $n = k(r-1) + 1$

here, $r \le \lceil \frac{n}{k} \rceil$

This is reversed statement of Pigeon Hole statements