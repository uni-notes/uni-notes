## Bayes’ Theorem

$$
P(h | D) = \frac{P(D|h) \cdot P(h)}{P(D)}
$$

| Hypothesis                       |                                                              |                                                             |
| -------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Maximum Likelihood               | the hypothesis (or class) that best explains the training data | $h_\text{ML} = \underset{h_i \in H}{\arg \max} \ P(D |h_i)$ |
| Maximum A Posteriori Probability |                                                              | $h_\text{MAP} = somethign$                                  |

$\arg \max$ is like maximum of a list

### Disadvantage
We need to calculate a lot of probabilities

## Bayes Optimal Classifier

Given new instance $x$

Consider $v=\{v_1, v_2 \}=\{\oplus, \ominus \}$

The optimal classifier is given by

$$
\underset{v_j \in V} {\arg \max}
\sum_{h_i \in H} \textcolor{hotpink}{P(v_j | h_i)} \ P(h_i | D)
$$

### Disadvantage

Very costly to implement. We need to calculate a lot of probabilities

## Gibbs Algorithm

Consider we have multiple independent hypotheses

1. Choose one hypothesis at random, according to $P(h|D)$
1. Use this to classify new instance

### Disadvantage

Lower accuracy

One more point in slide

## Naive Bayes
Already taught

## Bayesian Belief Network

### Independent event

Events $A, B, C, \dots$ are independent $\iff$

$$
P(A, B, C, \dots) = P(A) \times P(B) \times P(C) \times \dots

$$
