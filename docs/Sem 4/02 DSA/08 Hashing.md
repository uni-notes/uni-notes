Data structure to store key-element pairs. Each key-element pair is called an item.

### Benefits

1. data encryption
2. Search optimization, by reducing the search space

### Parts

1. Hash Function
2. Bucket/Array (called dictionary/table)

## Types

|  Hashing   |   Hash Function    |       Use        |                                                   |
| :--------: | :----------------: | :--------------: | ------------------------------------------------- |
| Component  |     $\sum x_n$     |                  |                                                   |
| Polynomial | $\sum x_{n} a^{n}$ |   Unique code    |                                                   |
|  Division  |      $k \% n$      | Reduce code size | $k=$ key (element)<br />$n =$ prime (unique code) |
|    MAD     |  $(a k + b) \% n$  |                  |                                                   |

## Finding Remainder in calculator

Mode > Bases > Decimal

Remainder
= Dividend - (Divisor * Quotient)
= Dividend - (Divisor * $\frac{\text{Dividend}}{\text{Divisor}}$)

### Reason

Dividend = (Divisor * Quotient) + Remainder

### Example

$$
\begin{align}
&  10 \% 3 \\&= 10 - \left( 3 * \frac{10}{3} \right) \\&= 10 - ( 3 * 3 ) \\&= 10 - 9 \\&= 1
\end{align}
$$

