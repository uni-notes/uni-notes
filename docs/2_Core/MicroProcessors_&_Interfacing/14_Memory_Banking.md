## Components

### 2704

2704 is ROM chip

two has W
Inverted W is M
It also has o, so ROM

### LS138

$3 \times 8$ decoder

The other one will be ram chip
We don’t need the entire memory, so we instead use in different way.

|       |      |
| ----- | ---- |
| $O_0$ | ROM1 |
| $O_3$ | RAM1 |
| $O_4$ | RAM2 |

## Blah

| $O_0$ | $A_0$ | $\overline{BHE}$ | Even | Odd  |
| ----- | ----- | ---------------- | ---- | ---- |
| 0     | 0     | 0                | ✅    | ✅    |
| 0     | 0     | 1                | ✅    |      |
| 0     | 1     | 0                |      | ✅    |
| 0     | 1     | 1                |      |      |

It is active low.