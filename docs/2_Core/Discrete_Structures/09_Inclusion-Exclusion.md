## Principle of Inclusion-Exclusion

$n(A)$ can also be represented as $|A|$

## Basic Counting

#### Sum Rule

Let $A_1 , A_2, \dots, A_n$ be disjoint(mutually-exclusive) sets

$n (A_1 \cup A_2 \cup \dots \cup A_n) = n(A_1) + n(A_2) + \dots + n(A_n)$

OR operation

#### Product Rule

$n (A_1 \cap A_2 \cap \dots \cap A_n) = n(A_1) \times n(A_2) \times \dots \times n(A_n)$

AND operation

## Inclusion-Exclusion

- $n(A \cup B) = 1 \iff$ mutually-exhaustive
- $n(A \cap B) = 0 \iff$ mutually-exclusive

Formulae

1. $n(A \cup B) = n(A) + n(B) - n(A \cap B)$

2. $$
   \begin{aligned}
   n(A \cup B \cup C)&= n(A) + n(B) + n(C) \\
   & \qquad - n(A \cap B) - n(B \cap C) - n(A \cap C) \\   & \qquad + n(A \cap B \cap C)
   \end{aligned}
   $$

3. $A' = S - A$

4. Demorgan

   1. $(A \cup B)' = A' \cap B'$
   2. $(A \cap B)' = A' \cup B'$

$$
\begin{aligned}
|A'| &= |U| - |A| \\
|A-B| &= |A \cup B'| \\
&= |A| - |A\cup B| \\
|A \cap B \cap C'| &= |A \cap B| - |A \cap B \cap C| \\
|A \cap B' \cap C| &= |A \cap C| - |A \cap B \cap C| \\
|A \cap B' \cap C'| &= |B' \cap C'| - |A' \cap B' \cap C'|
\end{aligned}
$$

## Gen Principle

$$
\begin{aligned}
\| A_1 \cup A_2 \cup \ldots \cup A_n \|
&= S_1 - S_2 + S_3 - \ldots + (-1)^{n-1} S_n \\
&= \sum\limits_{i = 1}^n |A_i|
- \sum\limits_{i,j} |A_i \cap A_j|
+ \sum\limits_{i, j, k} |A_i \cap A_j \cap A_k| \\
& \qquad + \ldots
+ (-1)^{n-1} |A_1 \cap A_2 \cap \dots \cap A_n|
\end{aligned}
$$

## Selection

|             | Permutation                | Combination                               |
| ----------- | -------------------------- | ----------------------------------------- |
| ordered?    | Y                          | N                                         |
| with rep    | $n^r$                      | $V(n,r)$                                  |
| without rep | $nP_r = \frac{n!}{(n-r)!}$ | $nC_r = \frac{n!}{r!\ (n-r)!} = nC_{n-r}$ |

$nC_r = nP_r = 0 \iff n<r$

The no of $r$ combinations of $n$ distinct objects with unlimited repetitions

$$
\begin{aligned}
&= V(n,r) \\
&= (n-1+r)C_r &= (n-1+r)C_{n-1} \\
&= \frac{(n-1+r)!}{r! \ (n-1)!}
\end{aligned}
$$

Uses

- This is the no of ways of distributing $r$ similar balls into $n$ number boxes
- no of non-negative integer solutions of $x_1 + x_2 + \dots + x_n = r$
- no of binary nos with $(n-1)$ ones and $r$ zeros

## Integral Solutions

The no of non-negative integer solutions is given by $V(n,r)$

$$
\set{
x_1 a_1, x_2 a_2, \dots, x_n a_n
}
\iff
x_1 + x_2 + \dots + x_n = r
$$

## Derangement

special type of permutation of any $n$ objects such that **no** number takes it’s own place

$i_1, i_2, \dots, i_n \iff i_1 \ne 1, i_2 \ne 2, i_n \ne n$

normally, for any arrangement of $n$ numbers, no of arrangements = $n!$

$D_n =$ no of derangments possible for derangement of $n$ numbers

$$
\begin{aligned}
D_1 &= 0 \\
D_2 &= 1 \\
D_3 &= 2 \qquad \set{(3, 1, 2), (2, 3, 1)} \\
\vdots & \\
D_n &= n! \left[
1- \frac{1}{1!}  + \frac{1}{2!} - \frac{1}{3!} + \dots + (-1)^n \frac{1}{n!}
\right] \\
&= n! \left[
1 + \sum_{i=1}^n (-1)^i \frac{1}{i!}
\right]
\end{aligned}
$$

