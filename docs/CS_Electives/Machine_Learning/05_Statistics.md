Statistical concepts such as Parameter estimation, Bias, Variance help in the aspects of generalization, over-fitting and under-fitting

## Point Estimation

Best prediction of an interested quantity, which can be a

- single parameter
- vector of parameters
- whole function

## Point Estimator/Statistic

A good estimator $\hat \theta$ is a function whose output is close to the true underlying $\theta$ that generated the data

### Properties

|          | Expected deviation              |
| -------- | ------------------------------- |
| Bias     | from the true value             |
| Variance | caused by any particular sample |

## Function Estimation

Estimation of relationship b/w input & target variables, ie predict a variable $y$ given input $x$

$$
y = \hat y + \epsilon
$$

where $\epsilon$ is Bayes Error

## Theories

### Statistical Learning Theory

Helps understand performance when we observe only the training set, through assumptions about training and test sets

- Training/test data arise from same process
- i.i.d
  - Examples in each data set are independent
  - Training set and testing set are identically distributed
- We call the shared distribution, the data generating distribution $p_\text{data}$

But rarely used in practice with deep learning, as

- bounds are loose
- difficult to determine capacity of deep learning algorithms

### Probably Approximately Correct

[SLT](#Statistical-Learning-Theory) contradicts basic principles of logic, as according to logic

> To infer a rule describing every member of a set, one
> must have information about every member of the set

ML avoids this problem with probabilistic rules, by finding rules that are probably correct about most members of concerned set

### No-Free Lunch

[PAC](#Probably-Approximately-Correct) does not entirely resolve contradiction of logic.

No free lunch theorem states

> Averaged over all distributions, every algo has same error classifying unobserved points, ie no ML algo universally better than any other.

