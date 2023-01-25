# Decision Trees

> Entropy, as it relates to machine learning, is **a measure of the randomness in the information being processed**. The higher the entropy, the harder it is to draw any conclusions from that information. 

### Steps

1. Pick an independent variable

2. Find Entropy of all classes of that independent variable

$$
H(C_i) =
-P_\text{Pos} \log_2 (P_\text{Pos})
-P_\text{Neg} \log_2 (P_\text{Neg})
$$

3. Calculate gain of each independent variable for current set/subset of data

$$
\begin{aligned}
&\text{Gain}\Big(
\text{Value}(C_1), C_2
\Big) \\
=& H \Big(\text{Value}(C_1) \Big) \\
   & - \left[
\sum_{i=1}
\frac{n (C_2=\text{Value}_i)}{n \Big(\text{Value}(C_1) \Big)}
\times
H(C_2=\text{Value}_i)
\right]
\end{aligned}
$$

4. Pick the independent variable with the highest gain

5. Recursively repeat for each independent variable

### Advantage

✅ Decision tree traversal only checks the minimal subset of attributes that are required. Hence, in many situations, it skips checking a lot of attributes.

## KNN CLassifier

$k$-nearest neighbor

- Pick a value of $k$
- Calculate distance between unknown item from all other items
- Seect $k$ observations in the training data are nearest to the unknown data point
- Predict the response of the unknown data point using the most popular response value from $k$ nearest neighbors

Lazy Learning: It does not build models explicitly
KNN builds a model for each test element

### Value of $k$

|      | $k$ too Small      | $k$ too Large                                   |
| ---- | ------------------ | ----------------------------------------------- |
|      | Sensitive to noise | Neighborhood includes points from other classes |

### Distances

- Manhattan distance
- Euclidian distance
- Makowski distance

Refer to data mining distances

### Advantages

- No training period
- Easy to implement
- NEw data can be added any time
- Multi-class, not just binary classification

### Disadvantages

- We have to calculate the distance for all testing dataset, wrt all records of the training dataset
  - Does not work well with large dataset
  - Does not work well with high dimensional dataset
- Sensitive to noisy and mssing data
- Attributes need to scaled to prevent distance measures from being dominated by one of the attributes

## Components

| Component      | Meaning                                                      |
| -------------- | ------------------------------------------------------------ |
| Internal Nodes | Labelled by attributes names, to be tested on an attribute<br />Contain the splitting attributes |
| Leaf Nodes     | Labelled by class labels                                     |
| Edges          | determined by the nuo of outcomes on n attribute test conditions |

## Heuristics

Methods that help find optimal solution

## Decision Tree Approaches

| Approach      | Steps                                                        | Disadvantage                 |
| ------------- | ------------------------------------------------------------ | ---------------------------- |
| **Greedy**    | Divide the problem into different steps, take decisions at each step<br /><br />Build tree in Top-Down manner<br />Split train set into purer subsets (the best split) @ every node | Backtracking is not possible |
| **Recursive** |                                                              |                              |

## Hunt’s Algorithm

Let

- $t$ be a node
- $D_t$ be train dataset @ node $t$
- $y$ be set of class labels $\{ C_1, C_2, \dots, C_n \}$

### Steps

1. Make node $t$ into a leaf node. Label $t$ with class label $y_t$
2. Split using appropriate attribute
   1. Apply splitting criteria for each attribute and obtain impurity of split using that attribute
   2. Pick the attribute that gives the lowest impurity
   3. Label the node $t$ with this attribute
   4. Determine the outcomes
   5. Create nodes for each outcome
   6. Draw the edges
   7. Split the dataset
3. Repeat the steps for each subtree now

### Additional Cases

#### Empty Subset

Let’s say when splitting your data, you end up having an empty subset of the data

1. Make node $t$ into a leaf node
1. $t$ is labelled as the majority class of the parent dataset

#### All records have identical attribute values

1. Make $t$ into a leaf node
2. $t$ is labelled as the majority class represented in the current subset

#### Number of records fall below minimum threshold value

- Make $t$ into a leaf node
- $t$ is labelled as the majority class represented in the current subset

### Splitting Continuous Attributes

#### Binary Split

Choose a splitting value that results in a purer partition

#### Multi-Way Split

- Apply discretization
- Each bin will be a different node

```mermaid
flowchart LR

subgraph Multi-Way Split
direction TB
s2((Salary))

a2(( ))
b2(( ))
c2(( ))
d2(( ))
e2(( ))

s2 -->|< 10k| a2
s2 -->|10k - 30k| b2
s2 -->|30k - 50k| c2
s2 -->|50k - 80k| d2
s2 -->| > 80k| e2
end

subgraph Binary
direction TB
s1((Salary))

end
```

### Challenges

- How to split training records?
  - Find best splitting attribute (Test on Attribute)
  - Measure for goodness of split
- Stopping conditions
  Stop at situations that result in fully-grown tree (the tree has learnt all the important characteristics, which may not be optimal; in that case we may require some other stopping conditions)
    - When all records at a node have the same attribute value
    - When all records at a node have the same class label value
    - When a node receives an empty subset

## Attribute Test Condition and Outcomes

### Nominal

Binary split will have $2^{k-1} - 1$ combinations of the binary split, where $k=$ no of values. For eg:

```mermaid
flowchart LR

subgraph 1
direction TB
aa[Attribute] --> v1a[v1] & notv1a["Not v1 <br> (v2 or v3 or v4)"]
end

subgraph 2
direction TB
ab[Attribute] --> v1b[v1 or v2] & notv1b["Not (v1 or v2) <br> (v3 or v4)"]
end
```

## Missed this class

[Lec20.pdf](../assets/Lec20.pdf) 

## Terms

Variables

$$
\Delta = I(Parent) - \sum_{j=1}^k
\underbrace{
\frac{N(V_j)}{N} I (V_j)
}_\text{Weighted Average Impurity}
$$

- $I(\text{Parent}) =$ Impurity at parent
- $N =$ no of records before split (parent)
- $k =$ no of splits
- $V_j =$ denote a split
- $N(V_j) =$ no of records at split $V_j$
- $I(V_k) =$ impurity at child(split) $V_j$

## Steps to choose splitting attribute

- Compute degree of impurity of parent node (before splitting)
- Compute degree of impurity of child nodes (after splitting)
- Compute weighted average impurity of split
- Compute gain $(\Delta)$
  Larger the gain, better the test condition
- Choose attribute with largest gain
- Repeat for all attributes

### Note

==Information gain means the impurity measure is entropy==