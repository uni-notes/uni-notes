represents each record as a datapoint with $m$ dimensional space, where $m$ is the number of attributes

## KNN Classifier

$k$ Nearest Neighbor

Class label is the majority label of $k$ nearest neighbors

### Training

#### Requirements

- Set of labelled records
- **Normalized** Proximity/Distance metric
  - Min-Max normalization
  - $Z$-normalization
- Value of $k$ (preferably $k$ is odd)

### Steps

- Compute distance between test record and all train records
- Identify $k$ neighbors of test records
  (Low distance=high similarity)
- Use majority voiting to find the class label of test sample

### Choosing Value of $k$

| $k$       | Problems                                        |
| --------- | ----------------------------------------------- |
| too small | Overfitting<br />Susceptible to noise           |
| too large | Underfitting<br />Susceptible to far-off points |

1. Use a test set
2. Let $k = 3$
3. Record error rate of classifier/accuracy
4. $k = k+2$
5. Repeat steps 3 and 4, until value of $k$ for which
   1. error is min
   2. accuracy is maximum

### Regression

KNN can be used for regression as well

Predicted value will be the average of the continuous labels of $k$-nearest neighbors

## Distance-Weighted KNN Classifier

Closer points are given larger weights for the majority voting scheme