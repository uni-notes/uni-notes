## Multiplexer (MUX)

also called as data selector

Combinational circuit

Digital switch

- $2^n$ inputs
- 1 output
- $n$ selection lines

### Applications

1. servers
   1. multiple devices are connected to just a single server
2. telecommunication

### 2-1 Mux

- 2 inputs 
- 1 output
- 1 selection line

| S0   | M     |
| ---- | ----- |
| 0    |$i_0$ |
| 1    |$i_1$ |

$M = i_0 {s_0}' + i_1 s_0$

### 4-1 Mux

- 4 inputs
- 1 output
- 2 selection lines
| s0   | s1   | M     |
| ---- | ---- | ----- |
| 0    | 0    |$i_0$ |
| 0    | 1    |$i_1$ |
| 1    | 0    |$i_2$ |
| 1    | 1    |$i_3$ |

$Y = i_0 {s_0}' {s_1}' + i_1 {s_0}' s_1 + i_2 s_0 {s_1}' + i_3 s_0 s_1$

### 8-1 Mux

- 8 inputs
- 1 output
- 3 selection lines

### 16-1 Mux

- 16 inputs
- 1 output
- 4 selection lines

## Simplifying mux

1. draw truth table
2. Choose variable(s) as selection lines
3. other variable(s) as mux i/p
4. write the function in terms of the mux i/p
   (easier than drawing kmap)

## Building Mux using smaller

1. divide $n_1$ by $n_2$
2. no of muxes = sum of quotients

positions of s1 and s2 are important

MSD will be the selection line for the last mux

### 4x1 using 2-1

$$
n_\text{req} = 4 \\n_\text{available} = 2 \\
4/2 = 2, 2/2 = 1 \implies \text{no of muxes}= 2 + 1 = 3

$$

### 8x1 using 2-1

$$
n_\text{req} = 8 \\n_\text{available} = 2 \\
8/2 = 4, 4/2 = 2, 2/2 = 1 \implies \text{no of muxes}= 4 + 2 + 1 = 7

$$

### 8x1 using 4x1 and 2x1

two 4x1 and one 2x1

$8/4 = 2; 2/2 = 1$

### 16x1 using 4x1

$$
n_\text{req} = 16 \\n_\text{available} = 4 \\
16/4 = 4, 4/4 = 1 \implies \text{no of muxes}= 4 + 1 = 5

$$

### Diagram

![mux](img/mux.svg)

## De-multiplexer (De-mux)

it is a digital switch with

1. 1 input
2. $n$ selection lines
   1. determines which output is connected to the input
3. $2^n$ multiple outputs

### 1-2

| S0   | D0   | D1   |
| ---- | ---- | ---- |
| 0    | i    | 0    |
| 1    | 0    | i    |

$D_0 = i {S_0}', D_1 = iS_0$

### 1-4

| s0   | s1   | D0   | D1   | D2   | D3   |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | i    | 0    | 0    | 0    |
| 0    | 1    | 0    | i    | 0    | 0    |
| 1    | 0    | 0    | 0    | i    | 0    |
| 1    | 1    | 0    | 0    | 0    | i    |

$D_0 = i {s_0}'{s_1}', D_1 = i{s_0}'s_1, D_2 = i s_0 {s_1}', D_3 = i s_0 s_1$

### Diagram

![demux](img/demux.svg)

