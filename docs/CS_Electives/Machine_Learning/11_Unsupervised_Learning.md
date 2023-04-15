## Clustering

Assigning class label to set of unclassified items

### Labels

Labels for classes are unknown during clustering

We may label the clusters later

## K-Means Clustering

Input: $k =$ number of clusters

Output: Set of $k$ clusters

### Steps

1. Randomly choose $k$ objects from the dataset as the initial cluster centroids
   (centroid = center of a cluster)

2. For each of the objects
   1. Compute distance between current objects and $k$ cluster centroids
   2. Assign current object to that cluster to which it is closest
      
      If distance of a point between 2 clusters is same, then we assign the point to first centroid.

3. Compute ‘cluster centers’ $m$ of each cluster. These become the new cluster centroids
   
$$
\begin{aligned}
m_k &= \Big(\text{mean}(X), \text{mean}(Y) \Big) \\
X &= \text{List of } x \text{ coordinates} \\
Y &= \text{List of } y \text{ coordinates}
\end{aligned}
$$

4. Repeat steps 2-3 until [convergence criterion](#convergence-criterion) is satisfied

5. Stop

### Convergence Criterion

- Particular number of iterations
  We can derive this by testing and plotting graph of accuracy vs iterations
  
  or
  
- when clusters don’t change over subsequent iterations

### EM

EM = Expectation Maximization

## Gaussian EM K-Means Clustering

Probabilistic clustering which requires K means as well; the output is k means is fed into this model

### Gaussian

$$
P(C|x) \propto
\frac{1}{\sqrt{2\pi \sigma_c^2}} \exp \left(
\frac{something}{}
\right)
P(k)
$$

### Gaussian Mixture

$$
P(X) =
\sum_{k=1}^K \Pi_k \cdot  N(X | \mu_k, \Sigma_k)
$$

- $\mu_k =$ Means
- $\Sigma_k =$ Covariances
- $\pi_k =$ Mixing Coefficients
    - Proportion of each gaussian in the mixture

### EM for Gaussian Mixture

1. Initialize gaussian parameters $\mu_k, \Sigma_k, \pi_k$

   1. $\mu_k \leftarrow \mu_k$
   2. $\Sigma_k \leftarrow \text{cov $\Big($ cluster($k$) $\Big)$}$
   3. $\pi_k = \frac{\text{No of points in } k}{\text{Total no of points}}$

2. E Step: Assign each point $X_n$ an assignment score $\gamma(z_{nk})$ for each cluster $k$
   
$$
\gamma(z_{nk}) = \frac{
\pi_k N(x_n|\mu_k, \Sigma_k)
}{
\sum_{i=1}^K \pi_i N(x_n|\mu_i, \Sigma_i)
}
$$

3. M Step: Given scores, adjust $\mu_k, \pi_k, \Sigma_k$ for each cluster $k$
   
$$
\begin{aligned}
\text{Let }
N_k &= \sum_{n=1}^N \gamma(z_{nk}) \\   N &= \text{Sample Size} \\   
\\   
\mu_k^\text{new} &=
\frac{1}{N_k}
\sum_{n=1}^N \gamma(z_{nk}) x_n
\\   \Sigma_k^\text{new} &=
\frac{1}{N_k}
\sum_{n=1}^N \gamma(z_{nk})
(x_n - \mu_k^\text{new})
(x_n - \mu_k^\text{new})^T
\\   
\pi_k^\text{new} &= \frac{N_k}{N} \\   \end{aligned}
$$
   
4. Evaluate log likelihood
   
$$
\ln p(X| \mu, \Sigma, \pi) =
\sum_{n=1}^N
\ln \left|
\sum_{k=1}^K \pi_k N(x_n | \mu_k, \Sigma_k)
\right|
$$

5. Stop if likelihood/parameters converge

## K Means vs Gaussian Mixture

|                        | K means                                                      | Gaussian Mixture                                             |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                        | Classifier model                                             | Probability model                                            |
| Probabilistic?         | ❌                                                            | ✅                                                            |
| Classifier Type        | Hard                                                         | Soft                                                         |
| Pamater to fit to data | $\mu_k$                                                      | $\mu_k, \Sigma_k, \pi_k$                                     |
| ❌ Disadvantage         | If a class may belong to multiple clusters, we have to assign the first one<br />If sample size is too small, initial grouping determines clusters significantly | Complex                                                      |
| ✅ Advantage            | Simple<br />Fast and efficient, with $O(tkn),$ where<br />- $t =$ no of iterations<br/>- $k =$ no of clusters<br/>- $n =$ no of sample points | If a class may belong to multiple clusters, we have a probability to back it up |

## PCA

Principal Component Analysis

(LDA is labelled data, PCA is for unlabelled data)

### Steps

1. Same steps as LDA (Linear Discriminant Analysis)

2. Choose the best Principal Component

3. $$
   \begin{aligned}
   P_{ij} &= {\text{PC}_i}^T
   \begin{bmatrix}
   x_j - \bar x \\   y_j - \bar y
   \end{bmatrix} \\   
   i &= \text{Which PC we are using} \\   j &\in [1, n] \\   n &= \text{Sample Size}
   \end{aligned}
   $$

4. Now use this $P$ vector as the new reduced dimension feature

