Special case of transportation model, where no of supply nodes always = no of demand nodes

Input is a $n \times n$ matrix, where

- $n$ workers are assigned to $n$ jobs
- cells contain the value of cost associated with assignment

## Objectives

is one of the following

- minimize the total time to complete a set of jobs
- minimize cost of assignments
- maximize the skill ratings
- maximize total satisfaction of customers

## Assumptions

- Each machine/worker is assigned $\le 1$ job
- Each job is assigned to exactly 1 machine/worker

## Steps

1. Operations

   1. Row operation $R_i = R_i - \text{min} (R_i)$

   2. Col operation $C_i = C_i - \text{min} (C_i)$

      or 

   1. Col operation $C_i = C_i - \text{min} (C_i)$
   2. Row operation $R_i = R_i - \text{min} (R_i)$

2. Assign jobs to workers

   1. Only cells with value = 0 can be assigned
   2. Assignment of a cell must be unique

3. Cost of completion = Initial values of the assigned cells

## Special Cases

| Case                            | Method                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| No Unique Solution found        | 1. Draw minimum number of horizontal/vertical lines in the last reduced matrix, passing through all 0s<br/>2. Select smallest uncovered element<br/>3. Subtract it from every uncovered element<br/>4. Add it to every element at the intersection<br/>5. If no feasible solution, go to step 1<br/>6. Else, determine the optimal solution |
| Unbalanced Assignment           | Add rows/columns as required<br />Fill empty rows/columns with 0s |
| Maximization Assignment Problem | Multiple all cells with -1 (only for first operation)<br />**==Be careful of -ve sign==**<br />Final value = -1 x Total Cost |
| Disallowed Assignment           | If some cell is missing data, fill it in with $M$ (a very large number) |
