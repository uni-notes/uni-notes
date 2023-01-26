## Set

A set is a collection of elements

- $A \cup B$
- $A \cap B$
- $A - B = A \cap B'$

### Power Set

Set of all subsets

No of elements in subsets $|A| = n( \ P(A) \ ) = 2^n$ 

Always includes $\phi$

## Functions

If A has m elements and B has n elements, then the number of $f: A \to B$ is$n^m$, because each of the m elements can relate to n elements, so no of functions
$= \underbrace{ n \times n \times \dots}_{m \text{ times} } = n^m$

### Types

#### One-one (injective)

Every element has exactly one image

if $f(x)= f(y) \implies x=y$

if $x \ne y \rightarrow f(x) \ne f(y)$

#### Onto (surjective)

Range = codomain, ie every element of B should have a pre-image

x should be able to be expressed in terms of y
let $f(x) = y  \implies x = g(y)$

#### Bijective

a function that is both one-one and onto

### Properties

#### Domain

The set of values that the input can take, ensuring that the function is defined

#### Codomain

The set that the codomain is related to

#### Range

The set of values that the output can take, ensuring that the function is defined

### Composition

$g \circ f \ (x) = g(\ f(x) \ )$

Domain of $ g \circ f \ (x) = \set{x \in \text{domain } f | f(x) \in \text{domain } g}$

$g \circ f \ (x)$ not necessarily equal to $f \circ g \ (x)$

## Relation

A relation between two sets is a collection of ordered pairs containing one object from each set.

Consider a binary relation $R \sube A \times B$, where $A \times B = \{ (a,b)/ a \in A, b \in B \}$. If A and B contain $m$and $n$ elements respectively, then A x B contains $m \times n$ elements

Consider $R \sube A \times A$, where $A \times A = \{(a,b) / a \in A, b \in A \}$. If A has n elements, then A x A contains $n^2$ elements

### Complement of relation

if $R \sube A \times A,$ then its complement is $R' = \set{A \times A} - R$

## Properties of Relations

### Reflexivity

$(a, a) \in R, \forall a \in R$

Self loop at ***all*** vertices

### Irreflexivity

$(a,a) \in R, \forall a \in R$

Self loop at ***no*** vertex

### Symmetry

$(a,b) \in R \implies (b,a) \in R$

Requires self loops everywhere; otherwise it is not symmetric

### Asymmetry

$(a, b) \in R \implies (b,a) \notin R$

### Antisymmetric

- $(a, b) \in R, (b, a) \in R \implies a = b$

- $(a, b) \in R, a \ne b \implies (b,a) \notin R$ 

No pair of vertices are connected in both directions, and there are self-loops

### Transitive

$(a, b) \in R, (b, c) \in R \implies (a, c) \in R$

Asymmetry$\implies$ Anti-symmetry, but not vice-versa

## Notes

if R is asymmetric, it is irreflexive

if R is transitive and irreflexive, it is asymmetric

## Equivalence Relation

Relation which is

1. Reflexive
2. Symmetric
3. Transitive

or

1. reflexive
2. circular

### No of unique equivalence relations

- $n = 4 \to 15$
- $n = 5 \to 52$

15 and 52 are called bell numbers

## Equivalence Class

Equivalence relation R divides/partitions A into disjoint union of non-empty subsets called as equivalence classes

it is denoted by [any element of the main set]
$[x], x \in A$

naming is not unique

eg: [0], [1], [x], [y], [January]

$A= \set{1, 2, 3}, \quad R = \set{(1,1), (1,2), (2,3)}$
$[1] = \set{1, 2}, [2] = \set{3}$

### Properties

- $x \ R \ y \implies [x] = [y],$ even if $x \ne y$
- $x, y \in A \implies [x] = [y] \text{ or } [x] \cap [y] = \phi$

### Converse

If P is partition of A into non-empty disjoint subsets, then P is the set of equivalence classes for the equivalence relation E defined on A by 

$a \ R \ b \iff$ a and b belong to the same subset of P

## Examples of Equivalence Relation

### Congruence modulo

returns the remainder
basically `x % m` in programming (smallest result)

$x \equiv y(mod \ m), \text{ if } x = y + am, \quad  a, m \in \Z \\
\implies x \% m = y$

eg:$12 \equiv 2 (mod \ 5), -12 = 3(mod \ 5)$

#### Conclusions

1. $m$ divides $x-y$
2. $x \equiv y(mod \ m) \implies y \equiv x (mod \ m)$
3. $\equiv (mod \ m)$ divides $\Z$ into $m$ equivalence classes
   $\Z_m = [0], [1], [2], \dots, [m-1]$
   1. $[0]$ contains the set of all elements that return 0 as remainder, when divided by m
   2. $[m] = [0], [m+1] = [1], \dots$
4. $[x] + [y] = [x+y], [x][y] = [xy], -[x] = [-x]$

## Circular

A relation r on a set A is said to be circular if $(a,b) \in R \text{ and }(b,c) \in R \implies (c,a) \in R$

R is reflexive & circular $\iff$ R is an equivalence relation

## Operations on Relations

- $R_1 - R_2 = \{ (a,b)| (a,b) \in R_1 \text{ and } (a,b) \notin R_2 \}$
  $R_1 - R_2 \sube R_1$

- $R_1 \cup R_2 = \{ (a,b)| (a,b) \in R_1 \text{ or } (a,b) \in R_2 \}$

- $R_1 \cap R_2 = \{ (a,b)| (a,b) \in R_1 \text{ and } (a,b) \in R_2 \}$

- $R^{-1} = \{ (b,a) | (a,b) \in R \}$

### Notes

| Property of $R_1$ and $R_2$ alone | Property of $R_1 \cup R_2$ | Property of $R_1 \cup R_2$ |
| --------------------------------- | -------------------------- | -------------------------- |
| Reflexive and Symmetric           | same                       | same                       |
| Transitive                        | not necessary              | same                       |
| Equivalence                       | not necessary              | same                       |
| Anti-symmetric                    | not necessary              | same                       |
| Partial-ordering                  | not necessary              | same                       |

## Composition of Relations

Let $R \sube A \times B$ and $S \sube B \times C$

Then, the composition of $R$ and $S$ is $R \circ S = \{(x,z) | (x,y) \in R \text{ and } (y,z) \in S \}$

$R \circ S \ne S \circ R$

### Composition of relation on itself

- $R \circ R$ can be denoted by $R^2$
- $R^2 \circ R$ can be denoted by $R^3$
- $R^k \circ R^l = R^{k+l}, \quad k,l \ge 1$

#### Transitive Closure

$R^+ = R \cup R^2 \cup \dots \cup R^n$ 

$R^+$ is the smallest relation containing R that is transitive

#### Transitive Reflexive Closure

$R^* = R^+ \cup \set{ (a,a) | \textcolor{orange}{\forall} a \in A }$
(add all reflexive elements whether or not they exist in the relation)

$*$ is more than + so it is transitive **and** reflexive

#### Symmetric Closure

$R \cup R^{-1} = \set{(x,y), (y,x) \ | \ (x, y) \in \textcolor{orange}{R}, (y,x) \in \textcolor{orange}{R^{-1}} }$ 
(only add those symmetric elements that exist in the relation and its inverse)
