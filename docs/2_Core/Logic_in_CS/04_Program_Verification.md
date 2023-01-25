## Hoare Triple Notation

$\set{\phi} P \set{\psi}$

eg: $\set{x \ge 0} \text{ fact } \set{f=x!}$

## Correctness

Total Correctness $\implies$ Partial Correctness

|                  | Partial                                              | Total                                                |
| ---------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| loop termination | not neccessary                                       | necessary                                            |
| correctness      | $\models_\text{par} \set{\phi} P \set{\psi}$         | $\models_\text{tot} \set{\phi} P \set{\psi}$         |
| incorrectness    | $\not \models_\text{par} \set{\phi} P \set{\psi}$    | $\not \models_\text{tot} \set{\phi} P \set{\psi}$    |
| holds when       | $\vdash_\text{par} \set{\phi} P \set{\psi}$ is valid | $\vdash_\text{tot} \set{\phi} P \set{\psi}$ is valid |

eg:

$$
\begin{aligned}
\models_\text{tot} \set{x \ge 0} &\text{ fact } \set{f=x!} \\
\not \models_\text{tot} \set{\top} &\text{ fact } \set{f=x!} \text{(as the loop will not terminate)} \\
\models_\text{par} \set{x \ge 0} &\text{ fact } \set{f=x!} \\
\models_\text{par} \set{\top} &\text{ fact } \set{f=x!}
\end{aligned}
$$

## Proof Tableaux

$$
\begin{aligned}
& \top \\
& \qquad \phi_0 & (implied)\\
& C_1 \\
& \qquad \phi_1 & \text{(assignment)}\\
& C_2 \\
& \qquad \phi_2 & \text{(assignment)}\\
& \vdots \\
& C_{n-1} \\
& \qquad \phi_{n-1} & \text{(assignment)}\\
& C_n \\
& \qquad \phi_n & \text{(assignment)}\\
\end{aligned}
$$

justification could be

1. assigned
2. implied
