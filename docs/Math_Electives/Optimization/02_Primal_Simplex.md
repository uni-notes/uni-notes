Vertices can be determined algebraically, when all constraints RHS and variables are non-negative.

==**Always start a problem on the left-side sheet**==; else you’ll waste time flipping pages

## Optimality & Feasibility

|             | Max Prob                                 | Min Prob                                 |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| Feasibility | All basic vars $\ge 0$                   | Same as $\leftarrow$                          |
| Optimality  | Coeff of all basic vars in obj() $\ge 0$ | Coeff of all basic vars in obj() $\le 0$ |

## Direct Simplex Method

1. Make sure all RHS constraints are +ve. Else, multiply by -1 to change the sign
2. Convert the constraints inequality into equality

| Constraint | LHS represents                            | RHS represents                              | Difference                         |                                                      Example |
| :--------: | ----------------------------------------- | ------------------------------------------- | ---------------------------------- | -----------------------------------------------------------: |
|   $\le$    | usage of limited resources for activities | limit on the availability of resources      | Slack (unused) amount of resources | $4x + 3y \le 240$<br />$\implies 4x + 3y \textcolor{hotpink}{+ s_k} = 240$ |
|   $\ge$    | usage of limited resources for activities | minimum requirement of resource utilization | Surplus amount of resources        | $4x + 3y \le 240$<br />$\implies 4x + 3y \textcolor{hotpink}{- x_k} = 240$ |

3. Transpose obj()

   Example: $\max Z = 70x+50y \implies \max Z - 70x-50y=0$

4. Select Entering/Exiting vars
   
| Type         | Entering var<br />= Non-basic var with most ___ coefficient of obj() | Exiting var<br />= Basic var with  ___ ratio |
| ------------ | ------------------------------------------------------------ | -------------------------------------------- |
| Maximization | -ve<br />(entering causes fastest increase in value of obj()) | least                                        |
| Minimization | -ve<br />(entering causes fastest decrease in value of obj()) | least                                        |

5. Calculate pivot row
6. Check optimality is reached
   - if obj() coeff $\ge 0 \forall$ non-basic vars
   - Verify feasibility
   - **end here**
   
7. Else
   - Calculate other rows

   - Repeat steps 4-6

| Term          | Meaning/Formula                                              |
| ------------- | ------------------------------------------------------------ |
| Ratio         | $\text{Ratio} = \frac{\text{Solution for basic variable}}{\text{Constraint coeff for entering var}}$<br />==**Denominator > 0**==<br/>If constraint coeff $< 0$, ignore that case, and check other variables’ ratio |
| Pivot Element | Intersection of entering and exiting var                     |
| Pivot Row     | $\text{Pivot Row} = \frac{\text{Leaving Row}}{\text{Pivot Element}}$ |
| Other rows    | $\text{New row} = \text{Old row} - (\text{Coeff of var in pivot column} \times \text{Pivot row})$ |

If $x_3, x_4, \dots$ come in one equation each, treat them as basic vars. Use row operations to ensure 

|  Equation  | Coeff of $x_3, x_4$ |
| :--------: | :-----------------: |
| Constraint |          1          |
|   obj()    |          0          |

## Artificial Starting Solution

- Surplus var is not initially included in the list of basic vars
- If slack and artificial var are tied for leaving var, **artificial var leaves**

**Starting Steps**

1. If any constraints have negative RHS, multiply by -1
2. Convert to equality

| Constraint Sign | Introduce                                                    |
| :-------------: | ------------------------------------------------------------ |
|      $\le$      | + Slack var                                                  |
|       $=$       | + Artificial var $R_1, R_2, \dots$                           |
|      $\ge$      | - Surplus var (==subtraction==)<br />+ Artificial var $R_1, R_2, \dots$ |

### Big-M Method

| Step                                                         |       Maximization       |       Minmization        |
| ------------------------------------------------------------ | :----------------------: | :----------------------: |
| Introduce artificial var to **transposed** obj()<br />$M=$ **very large number**, say a million<br />For the sign (column on right), think of it as: making $R_1, R_2, \dots$ very anti-entering | $+ MR_1 + M R_2 \ \dots$ | $- MR_1 - M R_2 \ \dots$ |
| Make the table as usual                                      |                          |                          |
| Perform row transformation to eliminate $R_1, R_2, \dots$ in obj() row<br />(keeping basic rows the same) |    $Z - MR_1 - MR_2$     |    $Z + MR_1 + MR_2$     |
| Solve as usual                                               |                          |                          |

### Two-Phase Method

| Phase | obj()                                                |             Goal              |
| :---: | :--------------------------------------------------- | :---------------------------: |
|   1   | $\min R = \sum R_i$<br />(for both max/min problems) | Force artificial vars to be 0 |
|   2   | original LP’s obj()                                  |    Determine optimal soln     |

#### Phase 1 Steps

- Do transformation to make $R_i$ as 0 in $R$’s row; other rows unchanged
- Solve as usual

#### Phase 2 Steps

| Optimal value of<br />$R = \sum R_i$ | Artificial vars<br />in basis | $\implies$ |       Optimal Soln =        | Steps                                                        |
| :----------------------------------: | :---------------------------: | :--------: | :-------------------------: | ------------------------------------------------------------ |
|                $> 0$                 |                               |            |  ❌<br />(infeasible soln)   |                                                              |
|                 $=0$                 |              $0$              |            | Optimal soln of phase 2 LP  | - Drop columns in phase 1 table corr to artificial variables<br />- Use original obj() with constraints from optimal phase 1 table (ignore initial constraints). This gives phase 2 obj()<br />- Perform row transformation to eliminate phase 1 basic vars in phase 2 obj() row<br />- Solve as usual |
|                $= 0$                 |             $> 0$             |            | Optimal soln of original LP | - Drop all<br />&nbsp;&nbsp;- Non-basic artificial vars from optimal phase 1 table<br />&nbsp;&nbsp;- Variables from original prob with -ve coeff in row of optimal phase 1 table<br />- Solve as usual |

## Special Cases

|               Cases               |                  Meaning                  | Description                                                  | Identification in simplex                                    |
| :-------------------------------: | :---------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Degeneracy<br />(Degenerate soln) |       $\ge 1$ redundant constraints       | Nothing alarming<br />May lead to Cycling (Var enters & exits basis repeatedly w/o reaching optimality)<br />Can be temporary/permanent | Solution for one basic var $= 0$<br />Usually happens when there is a tie for the leaving variable |
|      Alternative<br />Optima      |       obj() \|\| binding constraint       | obj() will assume the same optimal value at more than one corner point, $\implies \exists$ alternative soln | Coeff of non-basic var $=0 \implies$ var enters basic vars & obj() will not change |
|        Unbounded<br />Soln        | Unbounded soln space in $\ge 1$ direction | Values of some decision vars can be inc indefinitely, w/o violating constraint(s) | - Entries for one/more non-basic var column $\le 0 \ \forall$ constraint rows<br />and<br />- Entry for same non-basic var in obj() row<br />$\le 0$ : max<br />$\ge 0$ : min |
|       No feasible<br />Soln       |          Inconsistent contraints          |                                                              | Artificial var(s) exist(s) in basis even after optimality    |

## Summary of Methods

|       Constraints        | Method to use                 |
| :----------------------: | ----------------------------- |
|         All $=$          | Solve algebraically           |
|        All $\le$         | Direct Simplex Method         |
|        All $\ge$         | Big-M Method / 2 Phase Method |
| Mixture of $\le, =, \ge$ | Big-M Method / 2 Phase Method |

