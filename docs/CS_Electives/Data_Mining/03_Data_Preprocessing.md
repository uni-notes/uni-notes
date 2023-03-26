## Preprocessing Techniques

You don’t have to apply all these; it depends. You have to first understand the dataset.

| Technique                    | Meaning                                                      | Advantage                                                    | Disadvantage                                                 |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Aggregation                  | Combining/Merge data objects/attributes<br />**Continuous**: Sum, mean, max, max, min, etc<br />**Discrete**: Mode, Summarization, Ignoring | - Low processing cost, space, time<br />- Higher view<br />- More stable | Losing details                                               |
| Sampling                     | Creating **representative** subset of a dataset, whose characteristics are similar to the original dataset |                                                              |                                                              |
| Dimensionality Reduction     | Mathematical algorithm resulting in a set of new combination of old attributes | Eliminate noise and unnecessary features<br />Better understandability<br />Reduce time, memory and other processing cost<br />Easier visualization | Getting the original data is not possible after transformation |
| Feature Subset Selection     | Removing irrelevant and redundant attributes                 | Same as ^^                                                   | Extra resources required                                     |
| Feature Creation             | Create new attributes that can capture multiple important features more efficiently |                                                              |                                                              |
| Discretization               | Convert continuous attribute into categorial/discrete (for classification) |                                                              |                                                              |
| Binarization<br />(Encoding) | Convert continuous/categorical attribute into binary (association mining) |                                                              |                                                              |
| Attribute Transformation     | Mathematical transformations                                 |                                                              |                                                              |

Feature selection and Dimensionality reduction are used for **biomarkers analysis**

## Types of Sampling

### Random Sampling

| Random Sampling     | Data object put back into original population? | Duplicates? |
| ------------------- | :--------------------------------------------: | :---------: |
| With replacement    |                       ✅                        |      ✅      |
| Without replacement |                       ❌                        |      ❌      |

#### Problem

It may lead to misclassification, as not all classes are represented proportionally in the sample.

### Stratified Sampling

Different types of objects/classes with different frequency are used in the sample.

Useful especially in imbalanced dataset, where all the classes have large variation in their counts.

Ensures all classes of the population are well-represented in the sample.

#### Steps

- Draw samples from each class
    - equal samples, or
    - proportional samples, using % of the total of all classes
    - Gives us imbalanced dataset
- Combine these samples into a larger sample

### Progressive/Adaptive Sampling

Useful when not sure about good sample size

Computationally-expensive

#### Steps

```mermaid
flowchart TB
s["Start with small sample (100-1000)"] -->
a[Apply data mining algorithm] -->
e[Evaluate results] -->|Increase Sample Size| s

e -->|Best Result Obtained| st[/Stop/]
```

![sample_evaluation](assets/sample_evaluation.svg){ loading=lazy }

### Data Augmentation

- Duplication
- Fit it a distribution

## Dimensionality Reduction Algorithms

### PCA

Principal Component Analysis

Useful for continuous-valued attributes

Uses concepts of linear algebra, such as matrices, eigen values, eigen vectors

The new features are called as **Principal Components**

- Linear combinations of original attributes
- Perpendicular to each other
- Capture maximum amount of variance of data

### SVD

Singular Value Decomposition

## Feature Selection

```mermaid
flowchart TB

Attributes -->
ss[Search Strategy] -->
sa[/Subset of Attributes/] -->
Evaluation -->
sc{Stopping Criterion Reached?} -->
|No| ss

sc -->
|Yes| sel[/Select Attributes/] -->
Something
```

### Brute Force Approach

Consider a set with $n$ attributes. Its power set contains $2^n$ sets. Ignoring $\phi$, we get $2^{n-1}$ sets.

**Steps**

- Evaluate the performance of all possible combinations of subsets
- Choose the subset of attributes which gives the best results

### Embedded Approach

The data mining algorithm itself performs the selection, without human intervention

Eg: A decision tree automatically chooses the best attributes at every level

Builds a model in the form of a tree

- Internal nodes = labelled with attributes
- Leaf nodes = class label

### Filter Approach

Independent of data mining algorithm

```mermaid
flowchart LR
o[(Original<br /> Feature Set)] -->
|Select<br /> Subset| r[(Reduced<br /> Feature Set)] -->
|Mining<br /> Algorithm| Result
```

eg: Select attributes whose evaluation criteria(pairwise correlation/Chi^2^, entory) is as high/low as possible

### Wrapper Approach

Use the data mining algorithm (capable of ranking importance of attributes) as a black box to find best subset of attributes

```mermaid
flowchart LR
o[(Original<br /> Feature Set)] -->
s[Select<br /> Subset] -->
dm[Mining<br /> Algorithm] -->
r[(Reduced<br /> Feature Set)] & s

subgraph Black Box["Ran n times"]
	s
	dm
end
```

## Feature Creation

- Feature extraction
- Mapping data to new space
    - Time series data $\to$ frequency domain
    - For eg, fourier transformation
- Feature construction
    - Construct features from pre-existing ones
    - Eg
    - Area = length * breadth
    - Density = mass/volume

## Discretization

1. Sort the data in ascending order

2. Generate

     - $n-1$ split points

     - $n$ bins $\to$ inclusive intervals (specified by the analyst)

Then convert using binarization. But, why?

### Types

|                   |            Equal-Width Binning             |                 Equal-Frequency Binning                 |
| ----------------- | :----------------------------------------: | :-----------------------------------------------------: |
| Analyst specifies |                 No of bins                 |          Frequency of data objects in each bin          |
| Width             | $\frac{\text{Max-Min}}{\text{No of bins}}$ |                                                         |
|                   |                                            | Make sure atleast $n-1$ bins have the correct frequency |

## Binarization 

|                                        |         Method 1         | One-Hot Encoding |
| -------------------------------------- | :----------------------: | :--------------: |
| For $m$ categories, we need ___ digits | $\lceil \log_2 m \rceil$ |       $m$        |
| No unusual relationship                |            ❌             |        ✅         |
| Fewer variables?                       |            ✅             |        ❌         |

## Attribute Tranformation

|                        |                             $x'$                             |                                                              |        Property         |
| ---------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ | :---------------------: |
| Simple                 |                      $x^2, \log x, \| x \|$                      |                                                              |                         |
| Min-Max Normalization  | $\frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$ | $\frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}} * ({\max}_{\text{new}} - {\min}_{\text{new}}) + {\min}_{\text{new}}$<br />I didn’t exactly understand this |     $0 \le x \le 1$     |
| Standard Normalization |                    $\frac{x-\mu}{\sigma}$                    |                                                              | $\mu' = 0, \sigma' = 1$ |

