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

| Attribute Type | Dissimilarity                                                | Similarity                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Nominal        | $\begin{cases} 0, & p=q \\
 1, &p \ne q \end{cases}$          | $\begin{cases} 1, & p=q \\ 0, &p \ne q \end{cases}$          |
| Ordinal        | $\dfrac{\| p-q \|}{n-1}$<br />Values mapped to integers: $[0, n-1]$, where $n$ is the no of values | $1- \dfrac{\| p-q \|}{n-1}$  |
| Interval/Ratio | $\|p-q\|$                                                      | $-d$ <br /> $\dfrac{1}{1+d}$ <br /> $1 - \dfrac{d-d_\text{min}}{d_\text{max}-d_\text{min}}$ |

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
\| a_k - b_k \|^r
\right)^{\frac{1}{r}}
$$

| $r$      | Type of Distance                                             |               $d(x, y)$               | Gives                  | Magnitude of Distance | Remarks                               |
| -------- | ------------------------------------------------------------ | :-----------------------------------: | ---------------------- | --------------------- | ------------------------------------- |
| 1        | City block<br />Manhattan<br />Taxicab<br />$L_1$ Norm       |      $\sum_{k=1}^n \| a_k - b_k \|$       | Distance along axes    | Maximum               |                                       |
| 2        | Euclidean<br />$L_2$ Norm                                    | $\sqrt{ \sum_{k=1}^n \| a_k - b_k \|^2 }$ | Perpendicular Distance | Shortest              | We need to standardize the data first |
| $\infty$ | Chebychev<br />Supremum<br />$L_{\max}$ norm<br />$L_\infty$ norm |          $\max (\| x_k - y_k \|)$          |                        | Medium                |                                       |

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
\| x \| \ \ \| y \|
}
\sum_{i=1}^n x_i y_i \\
&= x \cdot y \\
\| x \| &= \sqrt{\sum_{i=1}^n x_i^2}
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

