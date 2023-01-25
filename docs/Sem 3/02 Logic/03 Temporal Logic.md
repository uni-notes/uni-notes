## Temporal Logic

### Time

|          | Linear Time            | Branching Time |
| -------- | ---------------------- | -------------- |
|          | sequential             | conditional    |
|          | infinite straight line | infinite tree  |
| branches | N                      | Y              |

### Temporal Connectives

#### State Quantifiers

For both LTL and CTL

| Symbol | Meaning                                                      |
| -----: | ------------------------------------------------------------ |
|    $X$ | ne==X==t state                                               |
|    $F$ | some ==F==uture state, including the current state           |
|     XF | some future state, from the next state onwards               |
|    $G$ | all future states, including the current state (==G==lobally) |
|     XG | all future states, from the next state onwards               |
|    $U$ | ==U==ntil $(<)$<br />$\phi U \psi$ means that $\phi$ is true initially and then suddenly $\psi$ becomes true. anything after that doesn’t matter |
|    $R$ | ==R==elease<br />$\phi R \psi$ means that both $\phi$ and $\psi$ occur once together. anything after that doesn’t matter |
|    $W$ | ==W==eak-until $(\le)$<br />(not really sure)                |

#### Path Quantifiers

(only for CTL)
| Symbol | Meaning                 |
| -----: | ----------------------- |
|      A | for ==A==ll paths       |
|      E | there ==E==xists a path |

### Operator Precedence

1. Unary operators
2. Temporal binary operators
3. Non-temporal binary operators

## States

$$
\underbrace{s_0 
\underbrace{\to}_\text{transition}
s_1}_\text{path}
$$

State diagram 

$\mathcal{P}$ is the power set

### Paths ($\pi$)

$\pi^i$ is the path originating from state $s_i$

### Deadlock

state having no further transitions

### Removing deadlock

add a another state $s_d$ which has a self-loop

### State Diagram

``` mermaid
flowchart LR
		s0 --> s1 & s2
		s1 --> s1
```

### Unwinding

Representing a state diagram using a binary tree is called as unwinding

``` mermaid
flowchart TB
s0 --> s1 & s2
s1 --> s[s1]
```

## CTL Equivalences

|                        | Paths | States |
| ---------------------- | ----- | ------ |
| Universal Quantifier   | A     | G      |
| Existential Quantifier | E     | F      |

$$
\begin{align}
\lnot AF \phi &= EG \lnot \phi \\\lnot EF \phi &= AG \lnot \phi \\\lnot AX \phi &= EX \lnot \phi \\
AF \phi &= A[T \cup \phi] \\EF \phi &= E[T \cup \phi]
\end{align}
$$

### Adequate Sets

$$
\begin{align}
AX \phi &= \lnot EX  \lnot \phi \\AG &=\\\end{align}
$$

there are more
