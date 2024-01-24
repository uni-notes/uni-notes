# Learning Experience $E$

Deals with how we train the model

## Model

A functional mapping between input and output

|          | Parametric                                                   | Non-Parametric                              |
| -------- | ------------------------------------------------------------ | ------------------------------------------- |
|          | Learn a function described by a parameter whose size is finite & fixed before data is observed | Complexity is function of training set size |
| Examples | Linear Regression                                            | Nearest Neighbor                            |

## Learning Types

| Type                   | Meaning                                                      | Application  |
| ---------------------- | ------------------------------------------------------------ | ------------ |
| Supervised             | Uses labelled data, to derive a mapping between input examples and target variable. |              |
| Unsupervised           | Learning from unlabelled data                                |              |
| Semi-Supervised        | $\exists$ labelled data and large amount of unlabelled data.<br/>Label the unlabelled data using the labelled data.<br/><br/>For example, **love** is labelled as emotion, but **lovely** isn’t<br /><br />Cotraining, Semi-Supervised SVM |              |
| Lazy/Instance-Based    | Store the training examples instead of training explicit description of the target function.<br/><br/>Output of the learning algorithm for a new instance not only depends on it, but also on its neighbors.<br/><br/>The best algorithm is KNN (K-Nearest Neighbor) Algorithm.<br/><br/>Useful for recommender system. |              |
| Active                 | Learning system is allowed to choose the data from which it learns.<br />There exists a human annotator.<br/><br/>Useful for gene expression/cancer classification |              |
| Multiple Instance      | Weakly supervised learning where training instances are arranged in sets.<br/>Each set has a label, but the instances don’t |              |
| Transfer               | Reuse a pre-trained model as the starting point for a model on a new related task |              |
| Reinforcement Learning | Learning in realtime, from experience of interacting in the environment, without any fixed input dataset.<br />It is similar to a kid learning from experience.<br/><br/>Best algorithm is **Q-Learning algorithm**. | Game playing |
| Bayesian Learning      | Conditional-probabilistic learning tool<br/>Each observed training expmle can incrementally inc/dec the estimated probability that a hypothesis is correct.<br/><br/>Useful when there is chance of false positive.<br/>For eg: Covid +ve |              |
| Deep Learning          |                                                              |              |

## Hyperparameters

Parameters that affect the prediction of a model.

## Types of Learners

They are not adapted by the ML algo itself, but we can use nested learning, where other algorithms optimize the hyperparameter for the ML algo.

|                      | Eager Learner                                        | Lazy Learner                                 |
| -------------------- | ---------------------------------------------------- | -------------------------------------------- |
| During<br />Training | Learns relationship between class label & attributes | Stores traing records                        |
| During<br />Testing  |                                                      | Perform computations to classify test record |
| Example              | - Decision Tree<br />- Rule-Based Classifier         | - Nearest-neighbor classifier                |

## Number of Variables

|          | Univariate Regression   | Multi-Variate                     |
| -------- | ----------------------- | --------------------------------- |
| $\hat y$ | $f(X_1)$                | $f(X_1, X_2, \dots, X_n)$         |
| Equation | $\beta_0 + \beta_1 X_1$ | $\sum\limits_{i=0}^n \beta_i X_i$ |
| Best Fit | Straight line           | Place                             |

## Degree of Model

|          | Simple Linear Regression              | Curve-Fitting<br />Polynomial Linear Regression              | Curve-Fitting/<br />Non-Linear Regression                    |
| -------- | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Equation | $\sum\limits_{j=0}^k \beta_j X_j$     | $\sum \limits_{j=0}^k \sum\limits_{i=0}^n \beta_{ij} (X_j)^i$ | Any of the $\beta$ is not linear                             |
| Example  | $\beta_0 + \beta_1 X_1 + \beta_1 X_2$ | $\beta_0 + \beta_1 X_1 + \beta_1 X_1^2 + \beta_1 X_2^{10}$   | $\beta_0 + e^{\textcolor{hotpink}{\beta_1} X_1}$             |
| Best Fit | Straight line                         | Curve                                                        | Curve                                                        |
|          |                                       |                                                              | You can alternatively perform transformation to make your regression linear, but this isn’t best<br/>1. Your regression will minimize transformed errors, not your back-transformed errors (what actually matters). So the weights of errors will not be what is expected<br/>2. Transformed errors will be random, but your back-transformed errors (what actually matters) won’t be a random process |

The term linear refers to the linearity in the coefficients $\beta$s, not the predictors
