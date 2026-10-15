## BCD

Binary Coded Decimal

Each digit of decimal will be represented in 4bit binary

To convert a number > 9, we add 6 (tutorial)

Eg: 33 = 0011 0011

## Gray Code

Reflection/Unit distance code

Mirror the halfway horizontally

just a way to send information in a private method

### 1 Bit

0

1

### 2 Bit

0 0 = 0

0 1 = 1

1 1 = 2

1 0 = 3

### 3 Bit

0 0 0 = 0

0 0 1

0 1 1

0 1 0

1 1 0

1 1 1

1 0 1

1 0 0

### XOR gate Shortcut

odd one detector

#### Binary to Gray

1. convert into binary
2. do XOR of
   1. Bring 1st digit down as it is
   2. do XOR of adjacent elements

#### Gray to Binary

1. bring 1st digit down as it is
2. do XOR diagonally after that

## Error detection and code correction

Parity is a technique to convert codes with **even no of 1s** or **odd no of 1s** by adding an extra bit.

Parity Bit is added in MSD position

we add a 1, if the number violates the parity type

### Even parity generator

we need even no of 1s

value of binary with even no doesn't get affected

XOR gate

| a    | b    | c    | even parity | odd parity |
| ---- | ---- | ---- | ----------- | ---------- |
| 0    | 0    | 0    | 0           | 1          |
| 0    | 0    | 1    | 1           | 0          |
| 0    | 1    | 0    | 1           | 0          |
| 0    | 1    | 1    | 0           | 1          |
| 1    | 0    | 0    | 1           | 0          |
| 1    | 0    | 1    | 0           | 1          |
| 1    | 1    | 0    | 0           | 1          |
| 1    | 1    | 1    | 1           | 0          |

Eg

1. _ 0 0 1 0 0 1 (even no of 1s)
   0 0 0 1 0 0 1
2. _ 0 0 1 0 1 1 (odd no of 1s)
   1 0 0 1 0 1 1

Circuit contains 2 XOR gates, or we can just do with one 

### Even Parity Checker

Checking if the parity being sent along with the bits is correct or not

Circuit contains 3 XOR gates

If there is no error, c = 0
If there is error, then c = 1

| P    | X    | Y    | Z    | Checker<br />C |
| ---- | ---- | ---- | ---- | :------------: |
| 0    | 0    | 0    | 0    |       0        |
| 0    | 0    | 0    | 1    |       1        |
| 0    | 0    | 1    | 0    |       1        |
| 0    | 0    | 1    | 1    |       0        |
| 0    | 1    | 0    | 0    |       1        |
| 0    | 1    | 0    | 1    |       0        |
| 0    | 1    | 1    | 0    |       0        |
| 0    | 1    | 1    | 1    |       1        |
| 1    | 0    | 0    | 0    |       1        |
| 1    | 0    | 0    | 1    |       0        |
| 1    | 0    | 1    | 0    |       0        |
| 1    | 0    | 1    | 1    |       1        |
| 1    | 1    | 0    | 0    |       0        |
| 1    | 1    | 0    | 1    |       1        |
| 1    | 1    | 1    | 0    |       1        |
| 1    | 1    | 1    | 1    |       0        |

### Odd parity generator

Circuit contains 2 XOR gates and 1 final not gate

or XNOR gate

### Odd parity checker

## Disadvantage of odd and even parity

we cannot find out where the error is present

eg: 1001 is correct even parity
but if the other end receives 0101, it is still correct by even parity, but it's not the same value

## Hamming Code

$A(t, m)$

eg: A (7, 4) means 7 total bits, 4 message bits; this means there are 3 parity bits

given no $= m_1 m_2 \dots m_m$
(b = the bit)

Advantage

1. we can generate parity
2. check parity
3. if there is any error, we can correct

### No of Parities

$2^p \ge p + m + 1$

- p = no of parity bits
- m = no of message bits

final message will have (m+p) no of bits

### Position of parity

Parities are added in the place of 2 powers, ie, $2^0, 2^1, 2^2, 2^3, \dots$, ie

- 1st position
- 2nd position
- 4th position
- 8th position
- so on

#### Example

| 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| p1   | p2   | m1   | p4   | m2   | m3   | m4   |

### Value of Parity

For even parity

For 3 parities, create a table of 3bit input and wherever there is one is the positions of the parities

- $P_1 \to 1, 3, 5, 7$
- $P_2 \to 2, 3, 6, 7$
- $P_3 \to 4, 5, 6, 7$

For 4 parities, create a table of 4bit input and wherever there is one is the positions of the parities

- $P_1 \to 1, 3, 5, 7, 9$
- $P_2 \to$
- $P_3 \to$
- $P_4 \to$

1. A (7, 4) (total, message) is received as 1 0 1 0 1 1 1. Determine the correct code when even parity exists.
2. 1 0 0 1 1 0 1 0

## Verilog for Hamming

m1 = D0, m2 = D1,$m_n = D_{n-1}$ 

``` verilog
module hamming_code(t, m);
  input[3:0] m;
  output[6:0] t;
  wire p1, p2, p3;
  assign p1 = (m[0] ^ m[1] ^ m[3]); // 1 3 5 7
  assign p2 = (m[0] ^ m[2] ^ m[3]); // 2 3 6 7
  assign p3 = (m[1] ^ m[2] ^ m[3]); // 4 5 6 7
  
  assign t = {p1, p2, m[0], p3, m[1], m[2], m[3]};
```