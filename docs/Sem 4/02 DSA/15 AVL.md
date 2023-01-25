**Balanced** BST that reduce the worst-case time complexity from linear to logarithmic.

## Balance Factor

$$
BF =
\text{Height of left subtree} - \text{Height of right subtree}
$$

Leaves are always balanced, as they have a balanced factor of 0.

## Balanced Tree

Tree with [balanced factor](#balanced factor) $\{ -1, 0, +1 \}$

## Unbalanced $\overset{\text{rotation}}{\longrightarrow}$ Balanced

Rotation Mechanism

| Unbalanced | Type of Rotation |
| ---------- | ---------------- |
| LL         | RR               |
| RR         | LL               |
| LR         | LR               |
| RL         | RL               |

```mermaid
flowchart

subgraph Balanced
direction TB
p((2)) --- q((1)) & r((3))
end
subgraph LR Unbalanced
direction TB
a((3)) --- b((1)) & c(( ))

b --- d(( )) & e((2))
end
subgraph LL Unbalanced
direction TB
f((3)) --- g((2)) & h(( ))

g --- i((1)) & j(( ))
end

subgraph RR Unbalanced
direction TB
k((3)) --- l(( )) & m((4))

m --- n(( )) & o((5))
end

subgraph RL Unbalanced
direction TB
s((3)) --- t(( )) & u((5))

u --- v((4)) & w(( ))
end
```

## Time Complexity

|  Operation  |   Compexity   |
| :---------: | :-----------: |
| Restructure |    $O(1)$     |
|   Search    | $O(\log_2 n)$ |
|  Insertion  | $O(\log_2 n)$ |
|  Deletion   | $O(\log_2 n)$ |

