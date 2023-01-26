## Hardware

- ALU is a combinational circuit
- clock is for the frequency
- Address lines are uni

## Pin Diagram

Names

- DIP (Dual Inline Package)
- QFP (Quad Flag Pack)

### Multiplexer

Intel used multiplexer

- minimizes the area required for the chip
- reduces performance

## Modes

|            | Minimum | Maximum             |
| ---------- | ------- | ------------------- |
|            | MN/MX’  | MN/MX’              |
| Logic      | 1       | 0                   |
| Size       | Smaller | Larger              |
| Processors | Single  | Multiple            |
| Cost       | Cheaper | Expensive           |
|            |         | 8087 (co-Processor) |

## MAC operations

Multiplied and Accumulated

$AX+B$

## Cycle

1. Clock = T state
2. Machine - memory access
3. Instruction - instruction access + decoding + …

## Setup Time

the time before the clock high, during which the data must be setup, to avoid data corruption

## Hold Time

the time after the clock high, during which the data must be held, to avoid data corruption