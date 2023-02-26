## Adders

### Half-Adder

Combination circuit that performs arithmetic sum of 2 single bit binary

Limitation: Adding carry is not possible

| x    | y    | C    | S    |
| ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    |
| 0    | 1    | 0    | 1    |
| 1    | 0    | 0    | 1    |
| 1    | 1    | 1    | 0    |

$S = x \oplus y, C= xy$

``` verilog
module half_adder(s, c, a, b);
	input a, b;
  output s, c;
  
  xor(s, a, b);
  and(c, a, b);
endmodule
```

### Full Adder

Arithmetic sum of 3 bit binary

| x    | y    | c    | S    | C    |
| ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    |
| 0    | 0    | 1    | 1    | 0    |
| 0    | 1    | 0    | 1    | 0    |
| 0    | 1    | 1    | 0    | 1    |
| 1    | 0    | 0    | 1    | 0    |
| 1    | 0    | 1    | 0    | 1    |
| 1    | 1    | 0    | 0    | 1    |
| 1    | 1    | 1    | 1    | 1    |

- $S = x \oplus y \oplus c$
- $C = xy + c(x \oplus y)$

``` verilog
module fullAdder(sum, carry, a, b, c);
  input a, b, c;
  output sum, carry;
  wire p, q, r;
  
  xor(sum, a, b, c);
  
  and(p, a, b)
  xor(q, a, b);
  and(r, q, c);
  or(carry, p, r);
```

### Full adder using half adders

2 half adders

We need the algebraic expressions as full adder, but using 2 half adders

Verilog code in slide 12

### 4 bit binary adder

also called as

- 4 bit ripple adder
- 4 bit parallel adder

binary adder performs arithmetic sum of two binary nos

to add two n bit binary nos, n no of full adders are required

|      |      |      |      |          |
| ---: | ---: | ---: | ---: | -------: |
|   C3 |   C2 |   C1 |   C0 |$C_{in}$ |
|      |      |      |      |          |
|      |   A3 |   A2 |   A1 |       A0 |
|  (+) |   B3 |   B2 |   B1 |       B0 |
|      |      |      |      |          |
|      |   S3 |   S2 |   S1 |       S0 |

### Diagram

![adders](img/adders.svg)

## Subtractors

### Half Subtractor

| x    | y    | B    | D    |
| ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    |
| 0    | 1    | 1    | 1    |
| 1    | 0    | 0    | 1    |
| 1    | 1    | 0    | 0    |

- $D = x'y + xy' = x \oplus y$
- $B = x'y$

```verilog
module halfSub(D, B, x, y);
	input x, y;
  output D, B;
  
	assign D = x ^ y;
  assign B = ~x + y;
  
endmodule
```

### Full Subtractor

| x    | y    | b    | D    | B    |
| ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    |
| 0    | 0    | 1    | 1    | 1    |
| 0    | 1    | 0    | 1    | 1    |
| 0    | 1    | 1    | 0    | 1    |
| 1    | 0    | 0    | 1    | 0    |
| 1    | 0    | 1    | 0    | 0    |
| 1    | 1    | 0    | 0    | 0    |
| 1    | 1    | 1    | 1    | 1    |

- $D = x \oplus y \oplus b$
- $B = x'y + b(x \oplus y)'$

### Full Subtractor using half-subtractor

2 half-subtractors

### Binary Subtraction

Practically, binary subtraction is performed only in 2’s complement form. Therefore, subtractor circuit is not of much use.

$A - B = A + \underbrace{(-B)}_\text{2's complement}$

==The complement of binary can be obtained using XOR gate==

### Parallel subtractor

Binary subtractor using 2’s complement

### Diagram

![subtractors](img/subtractors.svg)

## Other

### 4 bit binary parallel adder/subtractor

$V = c_2 \oplus c_3$

V denotes the overflow

- addition
    - M = C~0~ = 0
    - no change to the inputs by the XOR gate
    - Eg: $B_0 \oplus 0 = B_0$
    - S = A + B + M
    = A + B + 0
    = A + B
    - V = 0 means no overflow
    - V = 1 means overflow
- subtraction
    - M = C~0~ = 1
    - the inputs will get complemented by the XOR gate
    - Eg:$B_0 \oplus 1 = {B_0}'$
    - S = A + 1s comp of B + C
    = A + 1s comp of B + 1
    = A + 2s comp of B
    - V = 0 means no overflow
    - V = 1 means overflow

### BCD Adder

- valid values are from 0-9
- 10-15 are invalid (and hence will be don’t care condition)

we use 8 4 2 1 coding method

max possible sum of 2 BCD digits$= \underbrace{1}_\text{carry} + 9 + 9 = 19$

- Sum <= 9 without carry
    - no correction is needed
    - 2+3 = 5
- Sum > 9 without carry
    - then add 6
    - 5+7 = 12
- Sum <= 9 with carry
    - then add 6
    - 8 + 8

We need two 4bit parallel adders

==Whenever the output is undefined, we have to consider that case as don’t care==

- for eg, for BCD, we take 10-15 places as don’t-care

if z8, z4, z2, z1, and k are the outputs of the first adder, then:

| Decimal | k    | z8   | z4   | z2   | z1   | Corrected Binary Sum | BCD Sum        |
| ------- | ---- | ---- | ---- | ---- | ---- | -------------------- | -------------- |
| 0       | 0    | 0    | 0    | 0    | 0    | No correction        | 0000           |
| …       |      |      |      |      |      | No correction        | Same as binary |
| 9       | 0    | 1    | 0    | 0    | 1    | No correction        | 1001           |
| 10      | 0    | 1    | 0    | 1    | 0    | + 0110               | 0001 0000      |
| …       |      |      |      |      |      | + 0110               |                |
| 16      | 1    | 0    | 0    | 0    | 0    | + 0110               | 0001 0110      |
| 17      | 1    | 0    | 0    | 0    | 1    | + 0110               | 0001 0111      |
| 18      | 1    | 0    | 0    | 1    | 0    | + 0110               | 0001 1000      |
| 19      | 1    | 0    | 0    | 1    | 1    | + 0110               | 0001 1001      |

$C = z_2 z_8 + z_4 z_8 + k$

### Circuit

![other adders](img/otherAdders.svg)