## Regular Language Types

| RL Description | Method       | Difficulty for<br />Human | Difficulty for<br />Computer                          |
| -------------- | ------------ | ------------------------- | ----------------------------------------------------- |
| Verbal         | Textual      | Easy                      | Difficult<br />(Natural Language Processing required) |
| Reg Ex         | Algebraic    | Slightly difficult        | Difficult                                             |
| NFA            | Diagrammatic | Medium                    | Medium                                                |
| DFA            | Diagrammatic | Difficult                 | Easy                                                  |

## Regular Expression

are algebraic expression to describe the same class of strings (which can be recognized with finite memory)

Denoted as $R$

### Atomic

- $R = 0$ means $\{0 \}$

### Composite

Atomic regex with operations
(in order of precedence)

- Kleene star ${}^*$
- concatenation $\cdot$
- union: represented with $\cup$ or $|$ or $+$

Think similar to BODMAS: ^$,\times , +$

**Other operators**

- $R^+ = RR^*$
- $R^k$ for k -fold concatenation