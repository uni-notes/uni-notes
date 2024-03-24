## Proximity Measures

- Similarity
- Dissimilarity
  - Distance measure (subclass)

### Range

May be

- $[0, 1], [0, 10], [0, 100]$
- $[0, \infty)$

## Types of Proximity Measures

### Similarity

For document, sparse data

- Jacard Similarity
- Cosine Similarity

### Dissimilarity

For continuous data

- Correlation
- Euclidean

## Transformations

We should be careful; first study the problem and apply only if it is logical to complete the operation

|                  Fixed Range $\to [0, 1]$                   | $[0, \infty) \to [0, 1]$ |
| :---------------------------------------------------------: | :----------------------: |
| $s' = \frac{s - s_\text{min}}{s_\text{max} - s_\text{min}}$ |   $d' = \frac{d}{1+d}$   |

## Something

| Attribute Type                                               | Dissimilarity                                                | Similarity                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Nominal                                                      | $\begin{cases} 0, & p=q \\                                   |                                                              |
| 1, &p \ne q \end{cases}$          | $\begin{cases} 1, & p=q \\ 0, &p \ne q \end{cases}$ |                                                              |                                                              |
| Ordinal                                                      | $\dfrac{\vert  p-q \vert}{n-1}$<br />Values mapped to integers: $[0, n-1]$, where $n$ is the no of values | $1- \dfrac{\vert  p-q  \vert}{n-1}$                                  |
| Interval/Ratio                                               | $\vert p-q \vert$                                                    | $-d$ <br /> $\dfrac{1}{1+d}$ <br /> $1 - \dfrac{d-d_\text{min}}{d_\text{max}-d_\text{min}}$ |

## Dissimilarity Matrix

**Symmetric** $n \times n$ matrix, which stores a collection of dissimilarities for all pairs of $n$ objects

- $d(2, 1) = d(1, 2)$

It gives the distance from every object to every other object

Something

Example

| Object<br />Identifier | Test 1 | Tets 2 | Test 3 |
| ---------------------- | ------ | ------ | ------ |
|                        |        |        |        |
|                        |        |        |        |
|                        |        |        |        |

Compute for test 2

|       |  1   |  2   |  3   |  4   |
| :---: | :--: | :--: | :--: | :--: |
| **1** |      |      |      |      |
| **2** |      |      |      |      |
| **3** |      |      |      |      |
| **4** |      |      |      |      |

## Distance between data objects

### Minkowski’s distance

Let

- $a, b$ be data objects
- $n$ be no of attributes
- $r$ be parameter

The distance between $x,y$ is

$$
d(a, b) =
\left(
\sum_{k=1}^n
\vert  a_k - b_k  \vert^r
\right)^{\frac{1}{r}}
$$

| $r$      | Type of Distance                                             |                     $d(x, y)$                     | Gives                  | Magnitude of Distance | Remarks                               |
| -------- | ------------------------------------------------------------ | :-----------------------------------------------: | ---------------------- | --------------------- | ------------------------------------- |
| 1        | City block<br />Manhattan<br />Taxicab<br />$L_1$ Norm       |      $\sum_{k=1}^n \vert  a_k - b_k  \vert$       | Distance along axes    | Maximum               |                                       |
| 2        | Euclidean<br />$L_2$ Norm                                    | $\sqrt{ \sum_{k=1}^n \vert  a_k - b_k  \vert^2 }$ | Perpendicular Distance | Shortest              | We need to standardize the data first |
| $\infty$ | Chebychev<br />Supremum<br />$L_{\max}$ norm<br />$L_\infty$ norm |         $\max (\vert  x_k - y_k  \vert )$         |                        | Medium                |                                       |
|          | Makowski                                                     |                                                   |                        |                       |                                       |

Also, we have squared euclidean distance, which is used sometimes

$$
d(x, y) =
\sum_{k=1}^n |a_k - b_k|^2
$$

## Properties of Distance Metrics

| Property              | Meaning                         |
| --------------------- | ------------------------------- |
| Non-negativity        | $d(a, b) = 0$                   |
| Symmetry              | $d(a, b) = d(b, a)$             |
| Triangular inequality | $d(a, c) \le d(a, b) + d(b, c)$ |

## Similarity between Binary Vector

$M_{00}$ shows how often do they come together; $p, q$ do not have 11 in the same attribute

### Simple Matching Coefficient

$$
\text{SMC}(p, q) =
\frac{
M_{00} + M_{11} (\text{Total no of matches})
}{
\text{Number of attributes}
}
$$

### Jaccard Coefficient

We ignore the similarities of $M_{00}$

$$
\text{JC}(p, q) =
\frac{M_{11}}{M_{11} + M_{01} + M_{10}}
$$

## Similarity between Document Vectors

### Cosine Similarity

$$
\begin{aligned}
\cos(x, y) &= \frac{
xy
}{
\vert  x \vert  \ \ \vert  y  \vert 
}
\sum_{i=1}^n x_i y_i \\
&= x \cdot y \\
\vert  x  \vert &= \sqrt{\sum_{i=1}^n x_i^2}
\end{aligned} 
$$

| $\cos (x, y)$ | Interpretation              |
| ------------- | --------------------------- |
| 1             | Similarity                  |
| 0             | No similarity/Dissimilarity |
| -1            | Dissimilarity               |

### Document Vector

Frequency of occurance of each term

$$
\cos(d_1, d_2) =
\frac{d_1 d_2}{
||d_1|| \ \ ||d_2||
}
\sum_{i=1}^n d_1 d_2
$$

### Tanimatto Coefficient/Extended Jaccard Coefficient

$$
T(p, q) =
\frac{
pq
}{
||p||^2 + ||q||^2 - pq
}
$$

## Correlation

Used for continuous attributes

### Pearson’s Correlation Coefficient ($r$)

Range = $[-1, +1]$

| $r$  |                      |
| ---- | -------------------- |
| $-1$ | High -ve correlation |
| $0$  | No correlation       |
| $+1$ | High +ve correlation |

$$
\begin{aligned}
r(x, y)
&= \frac{
\text{Covariance}(x, y)
}{
\text{STD(x) } \text{ STD(y)}
} \\
& = \frac{
\sigma_{xy}
}{
\sigma_x \sigma_y
}
\end{aligned}
$$

$$
\begin{aligned}
\sigma_{xy}
&= \frac{1}{n} \sum_{i=1}^n (x_i - \bar x)(y_i - \bar y) \\
\sigma_{x}
&= \sqrt{
\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar x)^2
} \\
\sigma_{y}
&= \sqrt{
\frac{1}{n-1} \sum_{i=1}^n (y_i - \bar y)^2
}
\end{aligned}
$$

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
\text{Let } N_k &= \sum_{n=1}^N \gamma(z_{nk}) \\
N &= \text{Sample Size} \\
\mu_k^\text{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n \\
\Sigma_k^\text{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^\text{new}) (x_n - \mu_k^\text{new})^T \\
\pi_k^\text{new} &= \frac{N_k}{N}
\end{aligned}
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
