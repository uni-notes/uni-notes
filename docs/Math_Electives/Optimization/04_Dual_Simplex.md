This is not the same as duality

|                | Problem starts at                                 | Successive iterations continue to be | __ obtained in last iteration |
| -------------- | ------------------------------------------------- | ------------------------------------ | ----------------------------- |
| Primal Simplex | basic feasible solution                           | feasible                             | optimality                    |
| Dual Simplex   | infeasible solution, which is better than optimal | optimal                              | feasibility                   |

## Steps

Obj() must satisfy optimality condition of regular simplex method

1. Change all equalities to $\le$

| Constraint Type | Change                                                       |
| :-------------: | ------------------------------------------------------------ |
|      $\le$      |                                                              |
|      $\ge$      | Multiply both sides by -1                                    |
|       $=$       | Convert into inequality of both types<br />$x_1 + x_2 = 1 \implies \quad x_1 + x_2 \le 1, \quad -x_1 - x_2 \le -1$ |

2. Add slack vars (there is no surplus/artificial vars for dual simplex)
3. Transpose obj()
4. Select Entering/Exiting vars


|                         | Exiting var<br />= Basic var with most ___ value | Entering var<br />= Non-basic var with  ___ [ratio](#ratio) |
| ----------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| Max/Min (same for both) | -ve                                              | least                                                       |

5. Calculate pivot row (same as primal)
6. Check feasibility is reached
   - If value of all basic vars $\ge 0$
   - Verify optimality (same as primal)
   - **end here**
7. Else
   - Calculate other rows (same as primal)
   - Repeat steps 4-6


## Ratio

$$
\text{Ratio } = \left| \frac{
	\text{Obj() coeffient}
}{
	\text{Constraint coeff for exiting var}
} \right|
\qquad (\text{only magnitude})
$$

==**Denominator < 0**==

If constraint coeff $\ge 0 \ \forall$ non-basic $x_j$, the problem has no feasible solution

