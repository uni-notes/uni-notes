## Bayes’ Theorem

$$
\underbrace{P(\theta | y)}_{\mathclap{\text{Posterior Distribution} \qquad}}
= \frac{
	\overbrace{P(y|\theta)}^{\mathclap{\text{Likelihood Function}\qquad \quad }}
	\times
	\overbrace{P(\theta)}^{\mathclap{\qquad \quad \text{Prior Distribution}}}
}{
	\underbrace{P(y)}_{\mathclap{\qquad \text{Normalizing constant}}}
}
$$

| Hypothesis                       |                                                              |                                                             |
| -------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| Maximum Likelihood               | the hypothesis (or class) that best explains the training data | $h_\text{ML} = \underset{h_i \in H}{\arg \max} \ P(D \vert  h_i)$ |
| Maximum A Posteriori Probability |                                                              | $h_\text{MAP} = something$                                  |

$\arg \max$ is like maximum of a list

### Disadvantage
We need to calculate a lot of probabilities

## Bayes Optimal Classifier

Given new instance $x$

Consider $v=\{v_1, v_2 \}=\{\oplus, \ominus \}$

The optimal classifier is given by

$$
\underset{v_j \in V}{\arg \max}
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



## IDK

![image-20240106143739721](./assets/image-20240106143739721.png)

## Bayesian Classifier

Called as ‘Naive’ classifier, due to following assumptions

- Empirically-proven
- Scales very well

## Bayesian Rule

$$
P(C | X) = \frac{
P(X|C) \times P(C)
}{
P(X)
}
$$

Posterior depends on

- Likelihood
- Prior

$$
\text{Posterior} =
\frac{
something
}{
something
}
$$

### MAP Rule

**M**aximum **A** **P**osterior

Helps us decide the class during test phase

Assign $x$ to $c^*$ if $P(C=c^* | X=x) > P(C=c|X=x)$

## Naive Bayes Classification

Calculate posterior probability, based on assumption that all input attributes are conditionally-independent

### Drawbacks

1. Doesn’t work for continuous independent variable
   1. We need to use Gaussian Classifier
2. Violation of Independence Assumption
3. Zero outlook
