## Flip Flops

flip fl==o==p; cl==o==ck

in latches with enabled, it is observed that inputs are recognized only when enabled is 1. Therefore, it is possible to replace enabled with a momentary pulse called clock so that the inputs can be recognized only for a specified time. Such a device is called a flipflop.

(c means clock)

FF are usually edge triggered

the below truth tables are for positive trigger

## SR Flip Flop

almost identical compared to SR Latch, just that there is a clock 

| c   | s   | r   | $Q_n$ | $Q_{n+1}$   |
| --- | --- | --- | ----- | ----------- |
| 0   | X   | X   | X     | No change   |
| 1   | 0   | 0   | X     | $Q_n$       |
| 1   | 0   | 1   | X     | 0 (reset)   |
| 1   | 1   | 0   | X     | 1 (set)     |
| 1   | 1   | 1   | X     | invalid (0) |

| s   | r   | $Q_n$ | $Q_{n+1}$ |
| --- | --- | ----- | --------- |
| 0   | 0   | 0     | 0         |
| 0   | 0   | 1     | 1         |
| 0   | 1   | 0     | 0         |
| 0   | 1   | 1     | 0         |
| 1   | 0   | 0     | 1         |
| 1   | 0   | 1     | 1         |
| 1   | 1   | 0     | X         |
| 1   | 1   | 1     | X         |

### Expression

$Q_{n+1} = S + R' Q_n$ (simplified using KMap)

## D-Flip Flop

also called as transparent flip flop

| c   | d   | $Q_n$ | $Q_{n+1}$ |
| --- | --- | ----- | --------- |
| 0   | X   | X     | No change |
| 1   | 0   | X     | 0         |
| 1   | 1   | X     | 1         |

| d   | $Q_n$ | $Q_{n+1}$ |
| --- | ----- | --------- |
| 0   | 1     | 0         |
| 0   | 0     | 0         |
| 1   | 0     | 1         |
| 1   | 1     | 1         |

### Expression

$Q_{n+1} = d$ (simplified using KMap)

## JK Flip Flop

SR replaced with JK

Output of 

- $Q'$ will be *another* input of first NAND gate
- $Q$ will be *another* input of second NAND gate

| c   | j   | k   | $Q_n$ | $Q_{n+1}$                           |
| --- | --- | --- | ----- | ----------------------------------- |
| 0   | X   | X   | X     | No change                           |
| 1   | 0   | 0   | X     | $Q_n$                               |
| 1   | 0   | 1   | X     | 0 (reset)                           |
| 1   | 1   | 0   | X     | 1 (set)                             |
| 1   | 1   | 1   | X     | $\overline{Q}_n$ (toggle condition) |

| j   | k   | $Q_n$ | $Q_{n+1}$ |
| --- | --- | ----- | --------- |
| 0   | 0   | 0     | 0         |
| 0   | 0   | 1     | 1         |
| 0   | 1   | 0     | 0         |
| 0   | 1   | 1     | 0         |
| 1   | 0   | 0     | 1         |
| 1   | 0   | 1     | 1         |
| 1   | 1   | 0     | 1         |
| 1   | 1   | 1     | 0         |

$Q_{n+1} = j Q'_n + k' Q_n$

## T Flip Flop

Toggle Flip Flop

similar to XOR gate

We only want 00 and 11

Remove J and K, add T

| c   | t   | $Q_n$ | $Q_{n+1}$                           |
| --- | --- | ----- | ----------------------------------- |
| 0   | X   | X     | No change                           |
| 1   | 0   | X     | $Q_n$                               |
| 1   | 1   | X     | $\overline{Q}_n$ (toggle condition) |

| t   | $Q_n$ | $Q_{n+1}$ |
| --- | ----- | --------- |
| 0   | 0     | 0         |
| 0   | 1     | 1         |
| 1   | 0     | 1         |
| 1   | 1     | 0         |

$Q_{n+1} = T \oplus Qn$

## FF with Preset and Reset/clear

### Active High

| P   | R   | FF Response |
| --- | --- | ----------- |
| 0   | 0   | Normal FF   |
| 0   | 1   | Q = 0       |
| 1   | 0   | Q = 1       |
| 1   | 1   | Not used    |

### Active Low

| P   | R   | FF Response |
| --- | --- | ----------- |
| 0   | 0   | Not used    |
| 0   | 1   | Q = 1       |
| 1   | 0   | Q = 0       |
| 1   | 1   | Normal FF   |

## Diagrams

![FlipFlops](img/flipflops.svg){ loading=lazy }

## Verilog

```verilog
module srff(q,qbar, s, r, c);
  input s, r, c;
  output q, qbar;
  wire nand1, nand2;

  nand(nand1, s, c);
  nand(nand2, r, c);

  nand(q, nand1, qbar);
  nand(qbar, nand2, q);
endmodule

module testbench;
  reg s, r, c; // reg means storage
  wire q, qbar;

  initial begin
    c = 1'b1;
    repeat(2) #200 c = ~c;
  end

  initial begin
    s = 1'b0;
    repeat(8) #25 s = ~s;
  end

  initial begin
    r = 1'b1;
    repeat(13) #15 r = ~r;
  end
endmodule
```

Dlatch using

### Blocking

```verilog
module dLatch(input d, c, output reg q, qbar);
  always @ (d, c);
    if(c) begin
      #4 q = d;
      #4 qbar = ~d;
    end
endmodule
```

### Non-Blocking

```verilog

```

## Timing Diagram

basically the graph thingy

you’ll do it obviously

![](img/timing.svg){ loading=lazy }

## Excitation Table

==one FF var will be reverse of the other==
helpful for JK and SR

helps us to perform flip-flop conversions

In regular truth tables, (j,k, Qn) are inputs and Qn+1 is output
in excitation table, Qn and Qn+1 are inputs and j and k are outputs

### JK FF

| j   | k   | $Q_n$ | $Q_{n+1}$        |
| --- | --- | ----- | ---------------- |
| 0   | 0   | X     | $Q_n$            |
| 0   | 1   | X     | 0                |
| 1   | 0   | X     | 1                |
| 1   | 1   | X     | $\overline{Q}_n$ |

| $Q_n$ | $Q_{n+1}$ | j   | k   |
| ----- | --------- | --- | --- |
| 0     | 0         | 0   | X   |
| 0     | 1         | 1   | X   |
| 1     | 0         | X   | 1   |
| 1     | 1         | X   | 0   |

### T FF

similar to XOR gate

| T   | $Q_n$ | $Q_{n+1}$ |
| --- | ----- | --------- |
| 0   | 0     | 0         |
| 0   | 1     | 1         |
| 1   | 0     | 1         |
| 1   | 1     | 0         |

| $Q_n$ | $Q_{n+1}$ | T   |
| ----- | --------- | --- |
| 0     | 0         | 0   |
| 0     | 1         | 1   |
| 1     | 0         | 1   |
| 1     | 1         | 0   |

## D FF

| D   | $Q_n$ | $Q_{n+1}$ |
| --- | ----- | --------- |
| 0   | 0     | 0         |
| 0   | 1     | 0         |
| 1   | 0     | 1         |
| 1   | 1     | 1         |

| $Q_n$ | $Q_{n+1}$ | D   |
| ----- | --------- | --- |
| 0     | 0         | 0   |
| 0     | 1         | 1   |
| 1     | 0         | 0   |
| 1     | 1         | 1   |

### SR FF

| s   | r   | $Q_n$ | $Q_{n+1}$   |
| --- | --- | ----- | ----------- |
| 0   | 0   | 0     | 0           |
| 0   | 0   | 1     | 1           |
| 0   | 1   | 0     | 0           |
| 0   | 1   | 1     | 0           |
| 1   | 0   | 0     | 1           |
| 1   | 0   | 1     | 1           |
| 1   | 1   | 0     | X (Invalid) |
| 1   | 1   | 1     | X (Invalid) |

| $Q_n$ | $Q_{n+1}$ | S   | R   |
| ----- | --------- | --- | --- |
| 0     | 0         | 0   | X   |
| 0     | 1         | 1   | 0   |
| 1     | 0         | 0   | 1   |
| 1     | 1         | X   | 0   |

## Conversion

1. Identify source and destination FF
2. Tables
   1. Draw Truth table for destination FF
   2. extend it (change X into 0 and 1)
   3. Draw excitation table of source FF
   4. Merge both the tables
3. Draw KMap which provides expression for conversion with
   1. inputs as
      1. source FF input vars
      2. $Q_n$
   2. output as dest FF input vars
4. Draw circuit according to KMap

### SR using D

$S = D, R = D'$

### SR using JK

$S = j Q'_n, R = k Q_n$
