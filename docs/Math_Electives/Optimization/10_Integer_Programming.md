## Integer Programming

1. Solve as usual
2. Branch & Bound
   - Split with the variable with larger decimal value


$$
\begin{aligned}
\nabla f(x_0) &= 0 \\
\frac{\partial f}{\partial x_i} &= 0, \forall i
\end{aligned}
$$

## Application: Sports Scheduling

- Objective: Maximize team preferences
- Decisions: Which teams should play each other each week
  - Decision variable can be binary or discrete


Steps

- Define binary variable $x_{ijk}$
  - team $i$
  - team $j$
  - week $K$
- If team $I$ plays team $J$ in week $K$
  - $x_{IJK}=1$
  - Else 0
- Constraints
  - Play other teams in the same division twice
    - $\sum x_{IJk} = 2, \forall k$, where $I$ and $J$ are in same divisions
  - Place teams in other divisions once
    - $\sum x_{IJk} = 1, \forall k$, where $I$ and $J$ are in different divisions
  - Play exactly one team each week
    - $\sum x_{IjK} = 1, \forall j$

 
