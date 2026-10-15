Dual problem is a linear programming prob derived from the primal (original) LP model. It is basically a transposed version of the primal problem. Optimal Soln of Primal Prob = Optimal Soln of Dual Prob

Dual Form is preferred when no of constraints increases

| Form        | Numerical work $\propto$ | Preferred for                     |
| ----------- | ------------------------ | --------------------------------- |
| **Simplex** | No of constraints        | Fewer constraints/<br />Many vars |
| **Dual**    | No of vars               | Fewer vars/<br />Many constraints |

## Conversion of Primal $\to$ Dual

### Initial Steps

- All constraints RHS & vars are $\ge 0$
- Introduce slack and surplus vars (don’t introduce artificial vars)
- Dual var is defined $\forall$ primal constraint
- Dual constraint is defined $\forall$ primal var (including slack and surplus vars)

| Primal obj() | $\implies$ | Dual obj() | Inequality of new Constraints |
| :----------: | :--------: | :--------: | :---------------------------: |
|     max      |            |    min     |             $\ge$             |
|     min      |            |    max     |             $\le$             |

$$
\text{Dual Obj()} = \sum \text{RHS constraints}
$$

$$
\begin{aligned}
&\text{Dual constraint j} \implies \\
& \small{\sum \text{Constraint coeff of } x_i \text{ <Inequality> } \text{Non-transposed Obj() coeff of } x_i}
\end{aligned}
$$

## Computations

|                                                              | Formula                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Primal constraints column in iteration $i$]<br />(Primal basic vars) | [Inverse in iteration $i$] x [Primal constraints col]        |
| Primal obj() coeff of $x_j$<br />(Primal non-basic vars)     | LHS of non-transposed dual constraint$_j$ - RHS of non-transposed dual constraint$_j$ |
| [Dual vars in iteration $i$]                                 | [RHS of basic vars in non-transposed dual constraint] x [Inverse in iteration $i$] |
| Inverse                                                      | [Inverse] = [Columns in optimal primal table corr to vars not in obj()] |

## Checking Optimality & Feasibility

|             | Max Prob                              | Min Prob                              |
| ----------- | ------------------------------------- | ------------------------------------- |
| Feasibility | All primal **basic** vars $\ge 0$     | Same as $\leftarrow$                       |
| Optimality  | All primal **non-basic** vars $\ge 0$ | All primal **non-basic** vars $\le 0$ |

