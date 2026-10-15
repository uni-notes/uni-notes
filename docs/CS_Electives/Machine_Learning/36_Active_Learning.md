# Active Learning

```mermaid
---
title: Active Learning
---
flowchart LR
1[Label<br/>small sample] -->
2[Train<br/>model] -->
3[Predict<br/>unlabelled] -->
4["Query Selection<br/>(Rank poor predictions)"] -->
1
```

## Query Selection Strategies

| Type                  | Strategy                    | Measure                                                   | Advantage        |
| --------------------- | --------------------------- | --------------------------------------------------------- | ---------------- |
| Heuristic             | Uncertainty Sampling        | Least confident<br />Smallest margin<br />Highest entropy |                  |
|                       | QBC (Query By Committee)    | Vote entropy of enseble models                            |                  |
|                       | Expected Model Change       | Gradient of loss function wrt parameters                  |                  |
|                       | Expected Error Reduction    |                                                           |                  |
|                       | Variance Reduction          |                                                           |                  |
|                       | Density Weighted Methods    | Average similarity to entire unlabelled pool of examples  | Outliers ignored |
| Approximate Posterior | Monte-Carlo Dropout         |                                                           |                  |
|                       | Stochastic Weight Modelling |                                                           |                  |

