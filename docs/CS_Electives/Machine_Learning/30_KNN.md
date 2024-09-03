# $k$ Nearest Neighbor

Represents each record as a datapoint with $m$ dimensional space, where $m$ is the number of attributes

## Requirements

- Set of labelled records
- **Normalized** Proximity/Distance metric
  - Min-Max normalization
  - $Z$-normalization
- Odd value of $k$

## Choosing $k$

Similar to bias-variance tradeoff
$$
\begin{aligned}
\text{Flexibility} &\propto \dfrac{1}{k} \\

\implies \text{Bias} &\propto k \\
\text{Variance} &\propto \dfrac{1}{k}
\end{aligned}
$$

### Value of $k$

| $k$       | Problems                                                                                         | Low Bias | Low Variance |
| --------- | ------------------------------------------------------------------------------------------------ | -------- | ------------ |
| too small | Overfitting<br />Susceptible to noise                                                            | ✅        | ❌            |
| too large | Underfitting<br />Susceptible to far-off points: Neighborhood includes points from other classes | ❌        | ✅            |

![knn_decision_boundary](./assets/knn_decision_boundary.png)

![knn_bias_variance_tradeoff](./assets/knn_bias_variance_tradeoff.png)

Finding optimal $k$
1. Use a test set
2. Let $k = 3$
3. Record error rate of regressor/classifier
4. $k = k+2$
5. Repeat steps 3 and 4, until value of $k$ for which
   1. error is min
   2. accuracy is maximum

## Types


|        | Classification                                                                                                                                                                                                | Regression                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Output | Class label is the majority label of $k$ nearest neighbors                                                                                                                                                    | Predicted value will be the average of the continuous labels of $k$-nearest neighbors |
| Steps  | - Compute distance between test record and all train records<br>- Identify $k$ neighbors of test records<br>  (Low distance=high similarity)<br>- Use majority voiting to find the class label of test sample |                                                                                       |
|        | ![knn](./assets/knn.png)                                                                                                                                                                                      |                                                                                       |

## Distance-Weighted KNN

Closer points are given larger weights for the majority voting scheme