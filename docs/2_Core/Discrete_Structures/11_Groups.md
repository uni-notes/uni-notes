## Order of an element

### For $+, \oplus$

Regular/modulo addition operator

$x \oplus y =$ ??

what positive number multiplied gives the product as the identity element

### For $\times, \otimes$

Regular/modulo multiplication operator

what number raised to gives the final answer as the identity element

### For Both

If $a \in G$, G is a group, then the order of $a$ is the order of the cyclic group

## COSETS & Lagrange’s Theorem

Let

- $G$ be a group
- $H$ be its subgroup
- $a \in G$
- $h \in H$

1. COSETS may be duplicated, but we are only concerned about disjoint COSETS
2. Union of disjoint COSETS will be $G$
3. no of elements in COSET = no of elements in $H$

|           | Left COSET                     | Right COSET                    |
| --------- | ------------------------------ | ------------------------------ |
| $+$       | $a + H = \set{a + h, h \in H}$ | $H + a = \set{h + a, h \in H}$ |
| $\oplus$  |                                |                                |
| $\times$  | $a H = \set{a \times h}$       | $Ha = \set{h \times a}$        |
| $\otimes$ |                                |                                |

### Theorems

If $b \in G, b \ne a$

1. $a \in H \iff aH = H$
2. $aH = bH \iff a^{-1} b \in H$
3. $a \in bH \iff a^{-1} \in H b^{-1}$
4. $a \in bH \iff aH = bH$

$^{-1}$ means inverse (could be additive or multiplicative inverse)

## Lagrange’s Theorem

Let $G$ be a finite group of order $n$ and $H$ be any subgroup of $G$. Then the order of H divides the order of $G$.

$[G:H]$ = index of H = no of distinct left COSETS of H in G

Let $r$ be index of H. Let $|G| = n,|H| = m$. Then $n = mr \implies \frac n m = r$. Clearly, $m$ divides $n$

### Application

A cyclic group can only have subgroups with no of elements which divides the 

eg: $(Z_7, \oplus_7)$ can only have subgroups having no of elements dividing $7$. So, it can either be $<1>$ or $<7>$.