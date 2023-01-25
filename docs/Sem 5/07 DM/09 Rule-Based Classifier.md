## Rule-Based Classifier

Knowledge about dataset is stored in the form of if-then rules, in a rule database $R$

## Rule

$$
\text{LHS} \to \text{RHS}
$$

|                 | LHS                                | RHS         |
| --------------- | ---------------------------------- | ----------- |
| Contains        | Condition/Conjunct with attributes | Class label |
| Alternate Names | Antecedant<br />Pre-Condition      | Consequent  |

If precondition of rule $r$ matches attributes of record $x$

- $r$ **covers** $x$
- $x$ **fires/triggers** $r$

## Quality of Classification Rule

Consider $r: A \to y$, where

- $r$ is the rule
- $A$ is the antecedent
- $D$ is the dataset
- $|A|$ is the number of records covered by rule
- $|D|$ is the total number of records

| Quality Measure                      |                                                            | Formula                   |
| ------------------------------------ | ---------------------------------------------------------- | ------------------------- |
| Coverage$(r)$                        | Fraction of records covered by rule                        | $\dfrac{|A|}{|D|}$        |
| Accuracy$(r)$<br />Confidence Factor | Fraction of records for which the rule correctly predicted | $\dfrac{|A \cap y|}{|A|}$ |

## Steps

1. Find rule(s) that match(es) antecedent of record
2. Next steps:

| Number of rules triggered | Rules have same class label | Steps                                                        |
| ------------------------- | :-------------------------: | ------------------------------------------------------------ |
| 0                         |             N/A             | Add [default rule](#default rule)<br />Fallback to [default class](#default class) |
| 1                         |             N/A             | Assign consequent of rule as class label of test record      |
| Multiple                  |              ✅              | Assign consequent of rules as class label of test record     |
| Multiple                  |              ❌              | - Use the highest-priority ordered rule (computationally-expensive for training)<br />or<br />- Use majority voting scheme using unordered rules<br />(computationally-expensive for testing) |

### Default Rule

$$
\underbrace{}_\text{Empty Antecedant}
\to
\underbrace{y_d}_\text{Default class}
$$

### Default Class

Majority class represented by records not covered by rules in rulebase

## Desired Propertes of Rule-Based Classifier

| Desired Property             | Meaning                                         |
| ---------------------------- | ----------------------------------------------- |
| Rules are Mutually-Exclusive | Only 1 rule is triggered for any record         |
| Rules are Exhaustive         | $\exist \ge 1$ rule(s) that covers every record |

## Types of Rules

| Ordered                                                    | Unordered   |
| ---------------------------------------------------------- | ----------- |
| Priority assigned based on<br />- coverage<br />- accuracy | No priority |

## I missed 15min

## Extraction from Decision Tree

One rule is created for each path from root leaf

Keep taking the edges and use ‘and’ as the conjuction

### Why?

Rules are easier to understand than larger trees

## Sequential Covering Algorithm

1. Start with empy decision list $R$, training records $E$, class $y$
2. Learn-One-Rule function is used to extract the best rule for class $y$ that covers the current set of training records
3. Remove training records covered by the rule
4. New rule is added to the bottom of the decision list $R$
5. Repeat Steps 2, 3, 4 until stopping criterion is met
6. Algorithm proceeds to generate rules for the next class

During rule extraction, all training records for class $y$ are considered to be +ve examples, while those that belong to other classes are considered to be -ve examples

Rule is desirable such that it covers most of the +ve examples and none/very few -ve examples

## Learn-One-Rule Function

### General-to-Specific

#### Initial Seed Rule

$$
\underbrace{\phantom{\text{Empty Antecedent}}}_\text{Empty Antecedent} \to y_0

$$

Keep refining this initial seed rule, by adding more conjucts

### Specific-to-General

$$
\text{1 example of } y_0 \to y_0

$$

Keep refining this initial seed rule, by removing conjucts

## Metrics

### Foil’ Information Gain

First order inductive learner

Higher value $\implies$ better rule

### Likelihood Ratio Statistic

Let

- $k$ be number of classes
- $f_i$ be observed frequency of class $i$ examples, that are covered by rule
- $e_i$ be expected frequency of rule that makes random predictions
    - Probability of $i$ x Number of records covered by rule

$$
\begin{align}
R
&= 2 \sum_{i=1}^k \  f_i \ \log_2 \left(\frac{f_i}{e_i} \right) \\
e_i
&= \text{Frac}_i \times n_i
\end{align}
$$

Higher value $\implies$ better rule

## Types of Rule-Based Classifier Algorithms

|                    | Direct                      | Indirect                    |
| ------------------ | --------------------------- | --------------------------- |
| Extract rules from | data directly               | other classification models |
| Example            | - Ripper<br/>- CN2<br/>- 1R | - C4.5 Rules                |
