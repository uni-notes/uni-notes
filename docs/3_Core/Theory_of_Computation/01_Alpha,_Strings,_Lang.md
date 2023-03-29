## Model of Computation

1. FSM(Finite State Machine)
   Has Memory/State
2. FSM with Stack
3. Turing Machine
   Read/Write capability

A problem that cannot be solved by a Turing Machine is **not computable**.

## Symbols

|    Symbol     | Meaning                      | Description                                                  |
| :-----------: | ---------------------------- | ------------------------------------------------------------ |
|   $\Sigma$    | Set of Alphabet              | Non-empty finite set of symbols<br />eg: $\{0, 1 \}$         |
|               | String                       | Finite sequence of zero/symbols                              |
|     $\| s \|$     | Length of string s           |                                                              |
|  $\Sigma^k$   | Set of strings of length $k$ |                                                              |
|  $\Sigma^1$   | Set of strings of length 1   | eg: $\{0, 1 \}$                                              |
|  $\epsilon$   | Empty String                 | “”                                                           |
| $\phi, \{ \}$ | Null set<br />(Empty)        | $\phi \ne \epsilon \ne \{\epsilon\}$                         |
|      $L$      | Language                     | Finite/countably-infinite set of strings over a finite alphabet |

## Operations

### On Strings

| Operation     | Representation                                               | Description                                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Concatenation | $S_1 \cdot S_2$                                              | $S_1$ followed by $S_2$                                      |
| Power         | $S^n$                                                        | Concatenation with itself for $n$ times                      |
| Closure       | $\begin{aligned} \Sigma^+ &= \Sigma^1 \cup \Sigma^2 \cup \dots \\ \Sigma^* &= \Sigma^0 \cup \Sigma^1 \cup \Sigma^2 \cup \dots \\ &=  \{ \epsilon \} \cup \Sigma^+ \end{aligned}$ | Union of **infinite** concatenation with itself<br />Also called as Kleene Closure/Star |

### On Languages

| Operation               | Representation                                               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Union                   | $L_1 \cup L_2$                                               | Union of both sets                                           |
| Intersection            | $L_1 \cap L_2$                                               | Intersection of both sets                                    |
| Concatenation           | $L_1 \cdot L_2$                                              |                                                              |
| Complement              | $\begin{aligned} &L'\\ = &\Sigma^* - L \end{aligned}$        | Opposite of defined language<br />Swap all accepting and rejecting states |
| Closure of languages    | $L^*$                                                        | Similar to that of string                                    |
| Power Set of $\Sigma^*$ | $\begin{aligned} P(S) &= 2^{\Sigma^*} \\ \| P(S) \| &= 2^{\|S\|} \end{aligned}$ | Set of all subsets of $\Sigma^*$                             |

$$
\begin{aligned}
L = \phi \implies
L^* &= \{ \epsilon \} \\
\text{How?} \implies
L^*
&= L^0 \cup L^1 \cup \dots \\&= \{ \epsilon \} \cup \phi \cup \dots \\&= \{ \epsilon \}
\end{aligned}
$$

