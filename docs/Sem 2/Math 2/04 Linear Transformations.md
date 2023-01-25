## Linear Transformations

Consider a linear transformation

$$
L: 
\underbrace{U}_{\text{Domain}}
\to
\underbrace{W}_{\text{Codomain}}
$$
### Properies

1. $L(O_u) = O_w$
2. $L(\vec u \oplus \vec v) = L(\vec u) + L(\vec v)$
3. $L(\alpha \odot u) = \alpha \cdot L(\vec u)$

### Tricks

A transformation is not Linear Transformation if

- Power $\ne$ 1 or 0
- there is modulus(absolute value)
- determinant

## Kernel

$$
S = \set{
\vec u: L(\vec u) = O_w
}
$$

Set of all input values

## Range

$$
S = \set{ L(\vec u) }
$$

Set of all output values

## Properties

| Property |         Condition          |
| :------: | :------------------------: |
| One-one  |    Kernel = $\set{O_u}$    |
|   Onto   | dim(range) = dim(codomain) |

## Dimension Theorem

$$
\text{
dim(range) + dim(kernel) = dim(U)
}
$$

