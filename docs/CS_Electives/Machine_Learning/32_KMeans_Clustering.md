# K-Means

Input: $k =$ number of clusters

Output: Set of $k$ clusters

## Steps

1. Normalize $X$

2. Randomly choose $k$ objects from the dataset as the initial cluster centroids
   (centroid = center of a cluster)

3. For each of the objects

   1. Compute distance between current objects and $k$ cluster centroids

   2. Assign current object to that cluster to which it is closest

      If distance of a point between 2 clusters is same, then we assign the point to first centroid.

4. Compute ‘cluster centers’ $m$ of each cluster. These become the new cluster centroids

$$
\begin{aligned}
m_k &= \Big(\text{mean}(X), \text{mean}(Y) \Big) \\
X &= \text{List of } x \text{ coordinates} \\
Y &= \text{List of } y \text{ coordinates}
\end{aligned}
$$

4. Repeat steps 2-3 until [convergence criterion](#convergence-criterion) is satisfied

5. Stop

## Convergence Criterion

- Particular number of iterations
  We can derive this by testing and plotting graph of accuracy vs iterations

  or

- when clusters don’t change over subsequent iterations

## Limitations

- Clustering stuck in local minima: convergence dependent on initialization
- Measuring clustering quality is hard & relies on heuristics
- Non-probabilistic: Cluster assignment is binary
