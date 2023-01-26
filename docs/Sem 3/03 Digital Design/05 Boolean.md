## Boolean Laws

| Law             | Formula                                                      |
| --------------- | ------------------------------------------------------------ |
| Complementation | $$\bar0 = 1 \\ \bar1 = 0 \\ \bar{\bar{x}} = x$$              |
| and             | $$x \cdot 1 = x \\ x \cdot 0 = 0 \\ x \cdot x = x \\ x \cdot \bar x = 0$$|
| or              | $$x + 0 = x \\ x + 1 = 1 \\ x + x = x \\ x + \bar x = 1$$    |
| commutative     | $$x + y = y + x \\ xy = yx$$                                 |
| Associative     | $$x+(y+z) = (x+y)+z \\ x(yz) = (xy) z$$                      |
| Distributive    | $$x(y+z) = xy + xz \\ x + yz = (x+y)(x+z)$$                  |
| Demorgan's      | $$\overline{x+y} = \bar x \cdot \bar y \\ \overline{x \cdot y} = \bar x + \bar y$$ |

## Duality Principle

We can obtain the dual of any boolean expression by

1. operators are interchanged
   1. and -> or
   2. or -> and
2. identity elements are inverted
   1. 1 -> 0
   2. 0 -> 1

## Boolean Functions

### SOP ($\Sigma$)

Sum of Product

Represented by NAND gate

$$
\begin{align}
f(a,b,c) &= ab + bc \\
g(a,b,c) &= a'b + b'c
\end{align}
$$

### POS ($\pi$)

Product of Sum

Represented by NOR gate

$$
\begin{align}
f(a,b,c) &= (a+b)(b+c) \\
g(a,b,c) &= (a'+b)(b'+c)
\end{align}
$$

## Canonical Form

### Literal

Each variable within a term of a Boolean expression.

### Minterms

SOP

$$
m_0 + m_1 + m_2 + \dots
$$

Minterm (0) is targeted $x' = 0, x = 1$, Minterms are wherever the output is 1

Denoted by m~0,1,2~

They are $2^n$ possible combinations of AND terms, n = no of literals

in AND terms, a literal is

- primed if its value is 0 (complemented)

- unprimed if its value is 1

so that the AND of all literals are always 1

#### 2 variable minterm

$$
2^2 = 4
$$

| x    | y    | Minterm | Notation |           |
| ---- | ---- | ------- | -------- | --------: |
| 0    | 0    | x'y'    | m~0~     | 0' 0' = 1 |
| 0    | 1    | x'y     | m~1~     |  0' 1 = 1 |
| 1    | 0    | xy'     | m~2~     |  1 0' = 1 |
| 1    | 1    | xy      | m~3~     |   1 1 = 1 |

#### 3 var minterm

$$
2^3 = 8
$$

|      |      |      |        |      |
| ---- | ---- | ---- | ------ | ---- |
| 0    | 0    | 0    | x'y'z' | m~0~ |
| 0    | 0    | 1    | x'y'z  | m~1~ |
| 0    | 1    | 0    | x'yz'  | m~2~ |
| 0    | 1    | 1    | x'yz   | m~3~ |
| 1    | 0    | 0    | xy'z'  | m~4~ |
| 1    | 0    | 1    | xy'z   | m~5~ |
| 1    | 1    | 0    | xyz'   | m~6~ |
| 1    | 1    | 1    | Xyz    | m~7~ |

### Maxterms

POS

$$
M_0 \cdot M_1 \cdot M_3 \cdot \dots
$$

Maxterm (1) is targeted $x' = 1, x = 0$, maxterms are wherever the output is 0

Denoted by M~0,1,2~

In maxterms, there are$2^n$ of OR terms

In OR terms, literal is

- Primed if value is 1
- unprimed if value is 0

so that all the OR of all literals is 0

| x    | y    | Maxterm | Notation |             |
| ---- | ---- | ------- | -------- | ----------: |
| 0    | 0    | x + y   | M~0~     |   0 + 0 = 0 |
| 0    | 1    | x + y'  | M~1~     |  0 + 1' = 0 |
| 1    | 0    | x' + y  | M~2~     |  1' + 0 = 0 |
| 1    | 1    | x' + y' | M~3~     | 1' + 1' = 0 |

#### 3 var minterm

### Statements

- Any given functions can be expressed in canonical form without using truth table
- For sum of minterms
    - insert sum of missing literal and its complement
    - AND operation b/w terms
    - expand
- For product of maxterms, insert product of missing literal and its complement with OR operation, and expand

basically,

$$
\begin{align}
 & \ xyz + xy \\
=& \ xyz + xy(z+z')
\end{align}
$$

## K-Map

Karnaugh map

==uses grey code==, as only 1 bit changes from one place to the next

pictorial form of truth table used to simplify Boolean functions

made up of squares
All the squares represent minterm, or all represent maxterm

### Minterm

Result will be SOP

### Maxterm

Result will be POS

### 2 Var KMap

### 3 Var KMap

### 4 Var KMap

## Simplification of Boolean using KMap

### Minterm

- Each square with a 1 is an implicant
- Combine adjacent implicants to form prime implicant

### Rules for KMap grouping

1. Group size can be in terms of $2^n$
2. Try to always group in the max size
3. In a group, there should be at least one minterm(1)/maxterm(0) which is not a part of any other group
   Otherwise, it will be a redundant group

## Don't care condition

represents undefined function

The don't care terms are represented as $X$

Consider the don't care terms as

- 1 for maxterm
- 0 for minterm

for minterm KMap, you can consider the don't care terms as 1
for maxterm KMap, you can consider the don't care terms as 0

Not all don't care terms are necessary to be grouped, but if inclusion leads to larger groups of minterms, then include them also to minimize the function

When grouping, make sure that is at least one real minterm/maxterm in every group

## IDK

- AOI - AND OR inverter - SOP
- OAI - OR AND inverter - POS
