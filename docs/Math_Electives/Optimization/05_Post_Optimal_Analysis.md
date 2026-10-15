Deals with situation of finding new solution in efficient way when parameters are changed

## Actions

| Existing Soln<br />Feasible? | Existing Soln<br />Optimal? | $\implies$ | Action                         |
| :--------------------------: | :-------------------------: | :--------: | ------------------------------ |
|              ✅               |              ✅              |            | No action                      |
|              ✅               |              ❌              |            | Use primal simplex             |
|              ❌               |              ✅              |            | Use dual simplex               |
|              ❌               |              ❌              |            | Use generalized simplex method |

## Change in Feasibility

| Change                             | Steps                                                        |
| ---------------------------------- | ------------------------------------------------------------ |
| RHS of constraint equation changes | 1. Check feasibility with inverse method<br/>2. Calc new obj(), with values from step 1<br />3. Update table<br/>4. Use dual simplex<br />5. Calc obj() when feasibility is maintained/obtained |
| Addition of new constraint(s)      | 1. Check feasibility by substituting existing values into new constraint<br />2. Introduce slack/surplus into equation<br />3. Introduce slack/surplus row & column into table <br/>4. Update introduced basic var row using below formula<br />5. Use dual simplex<br />6. Calc obj() when feasibility is maintained/obtained |

$$
\begin{aligned}
& \text{Updated row of introduced basic var} \\
& = \text{Initial row of introduced basic var} - \left( \sum \text{coeff}_i \times x_i \right)
\end{aligned}
$$

where

- $x_i =$ basic variables in new constraint
- coeff$_i$ = coeff of $x_i$ in new constraint

## Change in Optimality

Caused due to change in obj()

**Steps**

1. Find the dual var values, using new obj() coeff
2. Check optimality
3. If optimality is maintained, go to step 6
4. Update obj() row using
   - coeff found when checking optimality
   - original obj() value, using above latest coeff
5. Use primal simplex
6. Calculate latest solution in **new** obj()

## New Constraint

| Type      | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| Redundant | A new constraint that does not affect feasibility of an existing optimum solution. |
| Binding   | - A new constraint that affects feasibility of an existing optimum solution.<br />- Simplex table after incorporating the constraint |
