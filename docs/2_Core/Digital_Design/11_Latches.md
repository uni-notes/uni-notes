Latches are usually level triggered

## SR Latch

used to store 0s and 1s applied as 2 inputs called ‘set’ and ‘reset’

has 2 stable states

1. SET state - Q = 1, Q’ = 0
2. RESET state - Q = 0, Q’ = 1

can be constructed using

- 2 cross-coupled NOR gates, or
- 2 cross-coupled NAND gates

NOR-based and NAND-based are reverse of each other

### NOR-based

$Q_{n+1} = (r + Q'_n)', Q'_{n+1} = (s + Q_n)'$

#### Simplified Truth Table

| s   | r   | $Q_n$ | $Q_{n+1}$      |
| --- | --- | ----- | -------------- |
| 0   | 0   | X     | $Q_n$          |
| 0   | 1   | X     | 0 (reset)      |
| 1   | 0   | X     | 1 (set)        |
| 1   | 1   | X     | invalid (0, X) |

### NAND-Based

active-low input latch, because if any of the input is 0, output is 1

$Q_{n+1} = (s \cdot Q'n)', Q'_{n+1} = (r \cdot Q_n)'$

#### Simplified Truth Table

| s   | r   | $Q_n$ | $Q_{n+1}$         |
| --- | --- | ----- | ----------------- |
| 0   | 0   | X     | invalid (1, X)    |
| 0   | 1   | X     | 1 (set)           |
| 1   | 0   | X     | 0 (reset)         |
| 1   | 1   | X     | $Q_n$ (no change) |

## With Enabled

We use ==NAND== SR latch with enabled input

The truth table is similar to NOR Latch

| e   | s   | r   | $Q_n$ | $Q_{n+1}$      |
| --- | --- | --- | ----- | -------------- |
| 0   | X   | X   | X     | No change      |
| 1   | 0   | 0   | X     | $Q_n$          |
| 1   | 0   | 1   | X     | 0 (reset)      |
| 1   | 1   | 0   | X     | 1 (set)        |
| 1   | 1   | 1   | X     | invalid (0, X) |

$Q_{n+1} = (s' \cdot Q'n)', Q'_{n+1} = (r' \cdot Q_n)'$

## D-Latch

Also called as delay/transparent latch

In SR latch, when the inputs are compliment of each other, then the output is either set state or reset state

The complimentary input conditions can be achieved by adding an inverter to one of the inputs of SR Latch

Now, the SR Latch has a single input called D

$D \to S, D' \to R$

used as a base for storage device in digital systems

| e   | d   | $Q_n$ | $Q_{n+1}$ |
| --- | --- | ----- | --------- |
| 0   | X   | X     | No change |
| 1   | 0   | X     | 0         |
| 1   | 1   | X     | 1         |

$Q_{n+1} = (d' \cdot Q'n)', Q'_{n+1} = (d \cdot Q_n)'$
cuz $s = d \implies s' = d' \quad r = d' \implies r' = d$????

$Q_{n+1} = d$, right?

## Diagram

![latches](img/latches.svg){ loading=lazy }
