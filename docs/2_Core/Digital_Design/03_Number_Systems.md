## Number Systems

$d_{n-1} \dots d_2 \ d_1 \ d_0 . d_{-1} \ d_{-2}$

- $d_{n-1} \to d_0$ = integer part
- $d_{-1} \to \dots$ = fraction part
- $d_{n-1}$ = MSD (most significant digit)

- $d_0$ = LSD (least significant digit)

| Decimal | Binary | Octal           | Hexa            |
| ------- | ------ | --------------- | --------------- |
| 10      | 2      | 8               | 16              |
|         |        | Groups of 3bits | Groups of 4bits |

| Decimal | Binary | Octal | Hexa |
| ------: | -----: | ----: | ---: |
|       0 |      0 |     0 |    0 |
|       1 |      1 |     1 |    1 |
|       2 |     10 |     2 |    2 |
|       3 |     11 |     3 |    3 |
|       4 |    100 |     4 |    4 |
|       5 |    101 |     5 |    5 |
|       6 |    110 |     6 |    6 |
|       7 |    111 |     7 |    7 |
|       8 |   1000 |    10 |    8 |
|       9 |   1001 |    11 |    9 |
|      10 |   1010 |    12 |    A |
|      11 |   1011 |    13 |    B |
|      12 |   1100 |    14 |    C |
|      13 |   1101 |    15 |    D |
|      14 |   1110 |    16 |    E |
|      15 |   1111 |    17 |    F |

## Binary

Bit = each digit

Nibble = group of 4bits

Byte = group of 8bits

|         |     Unsigned     |                  Signed Magnitude                   |                      1's comp                       |                      2's comp                      |
| :-----: | :--------------: | :-------------------------------------------------: | :-------------------------------------------------: | :------------------------------------------------: |
|  Range  |$+(2^n - 1)$ |$-(2^{n-1} - 1) \longleftrightarrow +(2^{n-1} - 1)$ |$-(2^{n-1} - 1) \longleftrightarrow +(2^{n-1} - 1)$ |  $-(2^{n-1} - 1) \longleftrightarrow +2^{n-1}$    |
|   +ve   |     regular      |                  same as unsigned                   |                  same as unsigned                   |                  same as unsigned                  |
|   -ve   |        -         |                     invert MSD                      |          Bit-by-bit complement of unsigned          | 1. Bit-by-bit complement of unsigned<br />2. Add 1 |
| +ve MSD |        0         |                          0                          |                          0                          |                         0                          |
| -ve MSD |        -         |                          1                          |                          1                          |                         1                          |

``` verilog
// 1s comp
module ones(a,y);
  input[3:0] a; // 4bits (0 to 3)
  output[3:0] y;
  assign y = ~a;
endmodule

// 2s comp
module twos(a,y);
  input[3:0] a; // 4bits (0 to 3)
  output[3:0] y;
  assign y = (~a) + 1;
endmodule
```

## Decimal

## Octal

## Hexadecimal

