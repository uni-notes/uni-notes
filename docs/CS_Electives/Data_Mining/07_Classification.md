## Classification

Supervised learning technique, which works with labelled dataset

In a classication dataset, you have

- Independent attributes $<A_1, A_2, A_3>$
- Discrete Target attribute
- Tuples/records/data objects/sample instance
  Tuple $= (x, y)$
    - $x =$ Feature Vector
    - $y =$ Class Label

### Task

To build a model that maps a faeture set to the corresponding class label

### Algorithms

Decision Trees, Naive Bayes, Rule-based

### Types

|                      | Eager Learner                                        | Lazy Learner                                 |
| -------------------- | ---------------------------------------------------- | -------------------------------------------- |
| During<br />Training | Learns relationship between class label & attributes | Stores traing records                        |
| During<br />Testing  |                                                      | Perform computations to classify test record |
| Example              | - Decision Tree<br />- Rule-Based Classifier         | - Nearest-neighbor classifier                |

## Process

1. Training (Induction)
2. Validation
3. Testing (Deduction)

If a model performs well for training data as well as unseen testing data, the model has **generalization ability**

Validation Set helps understand if the hyperparameters of the model are appropriate, and check over-fitting and under-fitting

Train-Validation-Test set split is usually 80:10:10

## Errors

1. Training Error
2. Testing Error
3. Generalization

A good model has 

## Model Overfitting

## Confusion Matrix

$n \times n$ matrix, where $n$ is the number of classes

|      |                |
| ---- | -------------- |
| TP   | True Positive  |
| TN   | True Negative  |
| FP   | False Positive |
| FN   | False Negative |

Actual Total number of

- +ve samples = TP + FN
- -ve samples = FP + TN

## Classification Accuracy Metrics

To analyze the classification

| Metric                                              | Meaning                                                      |                           Formula                            |
| --------------------------------------------------- | ------------------------------------------------------------ | :----------------------------------------------------------: |
| **Accuracy**                                        | $\frac{\text{Correct Predictions}}{\text{No of predictions}}$ |      $\frac{\text{TP + TN}}{\text{TP + FP + TN + FN}}$       |
| **Error**                                           | $\frac{\text{Wrong Predictions}}{\text{No of predictions}}$  | $\begin{align} & 1 - \text{Accuracy} \\ &\frac{\text{FP + FN}}{\text{TP + FP + TN + FN}}\end{align}$ |
| **Recall**<br />Sensitivity<br />True Positive Rate | How many actual +ve values were correctly predicted as +ve   | $\frac{\textcolor{hotpink}{TP}}{\textcolor{hotpink}{TP} + \text{FN}}$ |
| **Precision**<br />Positive Predictive Value        | Out of actual +ve values, how many were correctly predicted as +ve | $\frac{\textcolor{hotpink}{TP}}{\textcolor{hotpink}{TP} + \text{FP}}$ |
| **Specificity**<br />True Negative Rate             | Out of actual -ve values, how many were correctly predicted as -ve | $\frac{\textcolor{hotpink}{TN}}{\textcolor{hotpink}{TN} + \text{FP}}$ |
| **F Score**<br />F~1~ Score<br />F-Measure          | Harmonic mean between precision and recall<br />Close to lower value | $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| FP Rate                                             | Out of the actual -ve, how many were misclassified as Positive | $\begin{align}\alpha &= \frac{\textcolor{hotpink}{FP}}{\textcolor{hotpink}{FP} + \text{TN}} \\ &= 1 - \text{Specificity} \end{align}$ |
| FN Rate                                             | Out of the actual +ve, how many were misclassified as Negative | $\begin{align}\beta &= \frac{\textcolor{hotpink}{FN}}{\textcolor{hotpink}{FN} + \text{TP}} \\ &= 1 - \text{Sensitivity} \end{align}$ |

## Binary Classification

2-class classification

![confusion_matrix_True_False_Positive_Negative](assets/confusion_matrix.png){ loading=lazy }

## Multi-Class Classification

### Confusion Matrix with respect to A

|      | A    | B    | C    |
| ---- | ---- | ---- | ---- |
| A    | TP   | FN   | FN   |
| B    | FP   | TN   | TN   |
| C    | FP   | TN   | TN   |

## Classification Report of `sklearn`

Same thing taught in ML

Weighted average is used for imbalanced dataset

## Missed this class

[4. CS F415 Data Mining Decision Trees.pptx](assets/4. CS F415 Data Mining Decision Trees.pptx) 

## $k$-fold Cross Validation

Each re

- Train on 

