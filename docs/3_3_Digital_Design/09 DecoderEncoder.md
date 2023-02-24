## Decoder

converts binary numbers into decimal

- from n input lines
- to $2^n$ unique output lines

Decoder maps the value of the input to the subscript ?? Idk the exact word

### Applications

1. 7 segment displays (parkings, counters, etc)
2. selection in memories
3. de-compressing files

## Active

output -not input- gets affected

|                | Active High | Active Low |
| -------------- | ----------- | ---------- |
| on ==output==  | 1           | 0          |
| off ==output== | 0           | 1          |
| gate           | AND         | NAND       |

## Enabled

basically the switch for the decoder

Apart from the regular inputs, there is another input called ‘enabled’, which takes values high/low

Controls whether the circuit is on/off

- enabled high - e = 1 enables the decoder
- enabled low - e = 0 enables the decoder

## 2 to 4 line decoder

also called ‘1 of 4 decoder’

$n = 2; 2^n = 4$

### Active High

| x    | y    | d0   | d1   | d2   | d3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 1    | 0    | 0    | 0    |
| 0    | 1    | 0    | 1    | 0    | 0    |
| 1    | 0    | 0    | 0    | 1    | 0    |
| 1    | 1    | 0    | 0    | 0    | 1    |

$d_0 = x'y', d_1 = x'y, d_2 = xy', d_3 = xy$

### Active low

| x    | y    | d0   | d1   | d2   | d3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 1    | 1    | 1    |
| 0    | 1    | 1    | 0    | 1    | 1    |
| 1    | 0    | 1    | 1    | 0    | 1    |
| 1    | 1    | 1    | 1    | 1    | 0    |

(Everything will be complemented)

$d_0 = (x'y')', d_1 = (x'y)', d_2 = (xy')', d_3 = (xy)'$

we could also use OR gate? $d_0 = x+y, d_1 = x+y', d_2 = x'+y, d_3 = x'+y'$

## Decoder with enabled

x means don’t-care

### Enabled High, Active High

| e    | x    | y    | d0   | d1   | d2   | d3   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | x    | x    | 0    | 0    | 0    | 0    |
| 1    | 0    | 0    | 1    | 0    | 0    | 0    |
| 1    | 0    | 1    | 0    | 1    | 0    | 0    |
| 1    | 1    | 0    | 0    | 0    | 1    | 0    |
| 1    | 1    | 1    | 0    | 0    | 0    | 1    |

$d_0 = ex'y', d_1 = ex'y, d_2 = exy', d_3 = exy$

### Enabled Low, Active High

| e    | x    | y    | d0   | d1   | d2   | d3   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | x    | x    | 0    | 0    | 0    | 0    |
| 0    | 0    | 0    | 1    | 0    | 0    | 0    |
| 0    | 0    | 1    | 0    | 1    | 0    | 0    |
| 0    | 1    | 0    | 0    | 0    | 1    | 0    |
| 0    | 1    | 1    | 0    | 0    | 0    | 1    |

$d_0 = e'x'y', d_1 = e'x'y, d_2 = e'xy', d_3 = e'xy$

## 3 to 8 line decoder

$n = 3, 2^n = 8$

also called as

- ‘1 of 8’ decoder
- binary to octal decoder (not a mistake - octal is a subspace of decimal)

| x    | y    | z    | d0   | d1   | d2   | d3   | d4   | d5   | d6   | d7   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 0    | 0    | 1    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    |
| 0    | 1    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    |
| 0    | 1    | 1    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    |
| 1    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    |
| 1    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    |
| 1    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    |
| 1    | 1    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    |

$d_0 = x'y'z', d_1 = x'y'z, d_2 = x'yz', d_3 = x'yz, d_4 = xy'z', d_5 = xy'z, d_6 = xyz', d_7 = xyz$

## 4-16 decoder using 3-8 decoder

requires

1. two 3-8 decoders
   $x,y,z$ as inputs

2. $w$ as enabled

     - 1 enabled low

     - 1 enabled high
| w    | x    | y    | z    | Output |
| ---- | ---- | ---- | ---- | ------ |
| 0    | 0    | 0    | 0    | d0     |
| 0    | 0    | 0    | 1    | d1     |
| 0    | 0    | 1    | 0    | d2     |
| 0    | 0    | 1    | 1    | d3     |
| 0    | 1    | 0    | 0    | d4     |
| 0    | 1    | 0    | 1    | d5     |
| 0    | 1    | 1    | 0    | d6     |
| 0    | 1    | 1    | 1    | d7     |
| 1    | 0    | 0    | 0    | d8     |
| 1    | 0    | 0    | 1    | d9     |
| 1    | 0    | 1    | 0    | d10    |
| 1    | 0    | 1    | 1    | d11    |
| 1    | 1    | 0    | 0    | d12    |
| 1    | 1    | 0    | 1    | d13    |
| 1    | 1    | 1    | 0    | d14    |
| 1    | 1    | 1    | 1    | d15    |

## 4-16 decoder using 2-4 decoders

$$
16/ \textcolor{orange}4 = 4, 
4/ \textcolor{orange} 1 = 1 \\
req = 4 + 1 = 5
$$

we need 5 decoders in total

idk exactly

![building](img/building.png)

## Diagram

![decoders](img/decoders.svg)

## Combinational Logic implementation using decoder

Example: full adder

$S = \sum(1,2,4,7), C = \sum(3,5,6,7)$

- 3 inputs
- 3-8 decoder
- 2 or gates

refer the [gates notes](02 Gates.md#Universal Gates) to use different gates

## Encoder

converts decimal numbers into binary

is a combination circuit that performs inverse operation of a decoder

has $2^n$ inputs and $n$ outputs

is used to minimize data (compress)

only 1 input will be high

## 4-2 encoder, active high

| D3   | D2   | D1   | D0   | E1   | E0   |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 1    | 0    | 0    |
| 0    | 0    | 1    | 0    | 0    | 1    |
| 0    | 1    | 0    | 0    | 1    | 0    |
| 1    | 0    | 0    | 0    | 1    | 1    |

$E_1 = D_2+D_3, E_0 = D_1 + D_3$

## 8-3 Encoder, Active High

| D7   | D6   | D5   | D4   | D3   | D2   | D1   | D0   | E2   | E1   | E0   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    |
| 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    |
| 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0    |
| 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 1    | 1    |
| 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 1    | 0    | 0    |
| 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 1    |
| 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 1    | 0    |
| 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 1    | 1    |

- $E_2 = D_4 + D_5 + D_6 + D_7$
- $E_1 = D_3 + D_4 + D_6 + D_7$
- $E_0 = D_1 + D_3 + D_5 + D_7$

## Valid Line

- $V=0$
    - output is invalid
    - inputs are inactive
    - the outputs are not inspected and hence the output will be don’t care condition
- $V=1$
    - output is valid
    - at least 1 input is active

## Priority Encoder

encoder that includes priority function

helps the encoder give preference to the highest 

### 4-2 encoder

order of preference will be$\underbrace{D_3}_\text{highest} > D_2 > D_1 > \underbrace{D_0}_\text{lowest}$

| D3   | D2   | D1   | D0   | E1   | E0   | V    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | X    | X    | 0    |
| 0    | 0    | 0    | 1    | 0    | 0    | 1    |
| 0    | 0    | 1    | X    | 0    | 1    | 1    |
| 0    | 1    | X    | X    | 1    | 0    | 1    |
| 1    | X    | X    | X    | 1    | 1    | 1    |

Because of don’t care condition, we have to consider 0s as well for the equations
(figure it out on your own)

- $E_1 = D_2{D_3}' + D_3$
- $E_0 = D_1{D_2}'{D_3}' + D_3$
- $V = D_0 + D_1 + D_2 + D_3$

### 8-3 Encoder

order of preference will be $\underbrace{D_7}_\text{highest} > \ldots > \underbrace{D_0}_\text{lowest}$

| D7   | D6   | D5   | D4   | D3   | D2   | D1   | D0   | E2   | E1   | E0   | V    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | X    | X    | X    | 0    |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    |
| 0    | 0    | 0    | 0    | 0    | 0    | 1    | X    | 0    | 0    | 1    | 1    |
| 0    | 0    | 0    | 0    | 0    | 1    | X    | X    | 0    | 1    | 0    | 1    |
| 0    | 0    | 0    | 0    | 1    | X    | X    | X    | 0    | 1    | 1    | 1    |
| 0    | 0    | 0    | 1    | X    | X    | X    | X    | 1    | 0    | 0    | 1    |
| 0    | 0    | 1    | X    | X    | X    | X    | X    | 1    | 0    | 1    | 1    |
| 0    | 1    | X    | X    | X    | X    | X    | X    | 1    | 1    | 0    | 1    |
| 1    | X    | X    | X    | X    | X    | X    | X    | 1    | 1    | 1    | 1    |
