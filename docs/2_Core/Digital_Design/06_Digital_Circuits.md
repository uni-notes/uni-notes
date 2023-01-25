## Digital circuits

- combinational circuits
- sequential circuits

|                                 | Combinational | Sequential               |
| ------------------------------- | ------------- | ------------------------ |
| output depends on               | input         | input<br />present state |
| storage                         | ❌            | ✅                      |
| memory                          | ❌            | ✅                      |
| Feedback<br />(recursive input) | ❌            | ✅                      |

## Applications of combinational circuits

1. Arithmetic and logic functions
   1. adder
   2. subtractor
   3. comparator
   4. PLD (Programmable Logic Device)
2. Data Transmission
   1. multiplexer
   2. de-multiplexer
   3. encoder
   4. decoder
3. code conversion
   1. Binary
   2. BCD
   3. 7-Segment

## Sequential Logic Circuit

output depends on present inputs and past outputs (feedback)

sequential circuit will have storage elements to store the past outputs so that they can be fed back to the input

therefore, sequential circuit can be expressed as a combinational circuit with storage and feedback element

``` mermaid
flowchart LR
Inputs --> c[Combinational Circuit] --> Outputs --> s[Storage Element] --> c
```

### States

- Present state

- Next state

### Examples

1. counters
2. shift registers
3. sequence detector
4. function generator

## Clock

is a periodic square pulse

has 3 features

1. rising(+ve) edge $( \to )$ $(\uparrow)$
   not ideal, but alright for trigger because it is only for a short duration, but requires power
2. level(neutral) edge —
   worst for trigger because large power required and duration is for $t$ seconds
3. falling(-ve) edge $(\to)$ with a bubble $(\downarrow)$
   best for trigger because it is only for a short duration and requires least power

always low logic design is the best as it requires the least power

## Types of Sequential Circuits

1. +ve trigger/sensitive
2. Level trigger/sensitive
3. -ve trigger/sensitive

## Storage elements

|              | Latches      | FlipFlops                |
| ------------ | ------------ | ------------------------ |
| Clock        | ❌            | ✅                        |
| Sync Type    | Asynchronous | Asynchronous/Synchronous |
| Trigger Type | Level        | Edge                     |

