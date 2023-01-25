Consider a 1st order inexact DE

$$
M(x, y) dx + N(x, y) dy = 0, \quad
(M_y \ne N_x)
$$

### **Steps**

1. Find $M_y - N_x$
2. You will get one of the following cases
   the simplification will give in terms of a **single** variable

|      |                  Case 1                   |                  Case 2                  |
| :--: | :---------------------------------------: | :--------------------------------------: |
|      | $\dfrac{M_y - N_x}{\color{orange}-M} = h(y)$ | $\dfrac{M_y - N_x}{\color{orange}N} = g(x)$ |
|  IF  |         $e^{\int h(y) \cdot dy}$          |         $e^{\int g(x) \cdot dx}$         |

3. Multiply both sides of equation:
   Inexact DE $\times$ IF $\to$ Exact DE
4. Then, use [Exact DE](02_Exact_DE.md) method

## Shortcut

- Try to get everything in terms of simple integrals like $dx, dy, d(xy),d(x+y)$.
- Then use exact DE formulae

This way we can avoid the IF step
