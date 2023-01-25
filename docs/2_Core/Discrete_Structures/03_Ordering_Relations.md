## Ordering Relations & Lattices

### Partially-ordered sets(POSETS)

A relation R defined on a set S is said to be partially ordered if it is

1. reflexive
2. Anti-symmetric (uni-directional)
3. Transitive

Then (S, R) is a POSET.

Eg: $(\mathbb{Z}, \le), (\mathbb{Z}, \ge), (\mathbb{Z}^+ , /), (P(S) , \subseteq)$

#### Notes

1. Any partial ordering relation is denoted by $\preceq$

2. if $a \prec b$ denotes that $a \preceq b$ but $a \ne b$

3. if R is a partial order relation on S, the R^-1^ is also a partial order relation on S, where $R^{-1} = \{ (b, a) | (a,b) \in R \}$
   (S, R^-1^) is called the dual of (S, R)

4. Let$a, b \in S$ where $(S, \preceq)$ is a POSET.
   a and b are said to be comparable, if either $a \preceq b$ or $b \preceq a$
   otherwise a and b are not comparable

   Eg: $(\mathbb{Z}^+, /)$
   (2, 10) , (16, 8) is comparable; but (2, 7) isn't comparable as neither 2 nor 7 can divide each other

## Totally ordered Set

$\preceq$ is a totally-ordered relation if every 2 elements of S is comparable, and S is a totally-ordered set

Eg: $(Z, \le), (D_8,/)$ (divisors of 8)

### Not totally ordered

eg: $(Z^+, /), (D_{12}, /)$ (divisors of 12)

D~n~: set of all positive divisors of positive integer n
it is always a POSET, but not necessarily a TOSET

## Well Ordered Set

A relation that is

1. totally ordered
2. every subset of S has least element in the Hasse diagram
   doesn't have to be the least mathematically, such as the case of $(Z^-, \ge)$

WOSET $\implies$ TOSET
not all TOSETs are WOSETs, but all finite TOSETs are

Eg: $(N, \le), (Z^+, \le), (Z^-, \ge)$
$(Z^-, \le)$ is TOSET but not WOSET, as there is no subset ($- \infin$ is the least element)

## POSET/HASSE Diagram

1. draw from lower to upper direction
2. no loops
3. eliminate edges that are implied by transitiveness
4. no arrows

## Elements

### Minimal elements

indegree = 0 (excluding self)

a is a minimal element in S if there is no $b \in S$ such that $b \preceq a$

### Maximal elements

outdegree = 0 (excluding self)

a is a maximal element in S if there is no $b \in S$ such that $a \preceq b$

### Least element

Element a is called least element of S, if $a \preceq b, \forall b \in S$

The lowermost element of Hasse diagram

It has to be unique - ie the only minimal element

### Greatest Element

Element a is called greatest element of S if $b \preceq a, \forall b\in S$ 

The uppermost element of Hasse diagram

It has to be unique - ie the only maximal element

### Summary

Minimal and maximal points are end points that are related to ***all*** elements of the question set B, while least and greatest point are unique end point related to ***all*** elements of the question set B

## Bounds

Let A be a subset of S

==Bound is a set of the above elements==

### Upper bound

If u is an element of S such that $a \preceq u, \forall a \in A$, then u is called upper bound of A

should be related to both a and b

#### LUB/Supremum

Least upper bound

L==U==B/S==u==premum

### Lower bound

If $l$ is an element of S such that $l \preceq a, \forall a \in A$, then $l$ is called lower bound of A

should be related to both a and b

#### GLB/Infimum

Greatest lower bound

## Lattice

If the HASSE diagram starts and ends with a single point, it's called as a lattice.

A lattice is a POSET $(S, \preceq)$ in which each pair of elements has

1. LUB
   LUB of a and b: $a \lor b$ (join of {a, b}) -> sup(a, b)
2. GLB
   GUB of a and b: $a \land b$ (meet of {a,b}) -> inf(a,b)

eg: $(D_6, /)$

### Examples

$$
\left(P(S), \subseteq\right)\\
A, B \in P(S): A \lor B = A \cup B, A \land B = A \cap B
$$

$$
\left(P(S), \supseteq \right)\\
A, B \in P(S): A \lor B = A \cap B, A \land B = A \cup B
$$

$$
\left(P(S), \le\right)\\
A, B \in P(S): A \lor B = \text{max}(A,B), A \land B = \text{min}(A,B)
$$

$$
\left(D_n, / \right)\\
A, B \in D_n: A \lor B = \text{lcf}(A,B), A \land B = \text{hcf}(A,B)
$$

## Semi-Lattice

### Join Semi-Lattice

Lattice with only LUB

multiple starting points

Eg: $( \{2, 3, 60,180\},  /)$

### Meet Semi-Lattice

Lattice with only GLB

multiple ending points

Eg: $(I_{12}, /)$

## Properties of Lattices

Let $(L, \lor, \land)$ be an algebraic system defined by lattice $(L, \preceq)$

1. Idempotency
   1. $a \land a = a$
   2. $a \lor a = a$
2. Commutative
   1. $a \land b = b \and a$
   2. $a \lor b = b \lor a$
3. Associative
   1. $(a \land b) \land c = a \land (b \land c)$
   2. $(a \lor b) \lor c = a \lor (b \lor c)$
4. Absorption
   (opposite operation)
   1. $a \land (a \lor b) = a$
   2. $a \lor (a \land b) = a$
5. Distributive (not all lattices are distributable)
   1. $a \land (b \lor c) = (a \land b) \lor (a \land c)$
   2. $a \lor (b \land c) = (a \lor b) \land (a \lor c)$
6. Consistency
   $a \land b = a \text{ and } a \lor b = b$

