# Learning Theory

## Objectives of Learning

1. Ensure fit: $E_\text{in} \approx 0$
2. Ensure generalization: $E_\text{out} - E_\text{in} \approx 0$

## Hypothesis

Estimated model $\hat f(x) = \hat y$

### Hypothesis Set

Set of all hypotheses $H = \{ \hat f_i(x) \}$, both by

- machine
- human

## Good Characteristics of Hypotheses

### Parsimony

An explanation of the data should be made as simple as possible, but no simpler

### Occam’s Razor

The simplest model that fits the data is also the most plausible

Simple

- Complexity of $h$: MDL (Minimum Description Length)
- Complexity of $H$: Entropy, VC Dimension

$l$ bits specify $h$ $\implies$ $h$ is one of the $2^l$ elements of a set $H$

Exception - Looks complex but is actually simple: SVM

### Why is simpler better?

Simpler means out-of-sample performance

Fewer simple hypotheses than complex ones: $m_H(N)$ $\implies$ less likely to fit a given dataset: $m_h(N)/2^N$ $\implies$ more significant when it happens

### Falsifiability

If your data has chance of falsifying your assertion, then it does not provide any evidence for that assertion

Fit that means nothing: linear regression fit with just 2 data points

![image-20240421093701499](./assets/image-20240421093701499.png)

## Sampling Bias

If data is sampled in biased way, learning will produce a similarly biased outcome; problem for **both** causal and statistical learning

Non-Random Sampling

- non-representative sample that is not a random sample of the population we are interested in

  or

- study population is different from the target population

### Case Studies

- Presidential election results: Telephone: Truman vs Dewey
- Credit approval
- Creating portfolio based on long-term performance of currently-trading companies of S&P 500
  - You are looking at currently-trading stocks
  - Sampling bias caused by ‘snooping’
  - Solution: look & trade explicitly wrt S&P 500, not the comprising companies?

### Types

| Censoring                                                    | Truncation                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Given a random sample of individuals drawn from the population of interest, some variables – mainly the outcome – are observed only on individuals belonging to a subpopulation, while other variables are observed on all individuals in the sample | If all variables are observed only on individuals belonging to a subpopulation |
|                                                              | greater information loss than censoring                      |

### Solution: Matching distributions

Ensure that validation and test data matches the distribution of the true target population

The train and dev set need not match the same distribution, but it is recommended to sub-sample them such that it matches the target population

Doesn’t work for

- Region with $p(x)=0$ in-sample, but $p(x)>0$ out-of-sample

How? Gaussian estimation/Adversarial validation

1. Balancing using only train data
   1. Obtain probability $p$ for each datapoint belonging to the train data
   2. Weight these with $1/p$ to be sampled again

2. Distribution matching using target population
   1. Obtain probability $p$ for each datapoint belonging to the train data
   2. Weight these with $1/p$ to be sampled again


## Data Snooping

Also called p-hacking, specification search, data dredging, fishing

Process of trying a series of models until we obtain a satisfactory result, without accounting for such.

This will result in your model matching a particular dataset

This includes

- Parameters: Coefficients/Weights of the model
- Hyper-Parameters: Parameters that affect the learning of the model

It is possible to find a statistically significant result even if doesn’t exist, if you try hard enough

### Takeaways

- If a data set has affected any step in the learning process, its ability to assess the outcome of has been compromised. Hence it cannot be (fully) trusted in assessing the outcome.
- For a given problem type, if you perform an action which you would not do if the data were different, then you must penalize this action for generalization
- Using known properties of the target function does not have to be penalized, as it is not dataset-specific

### Solutions

- **Avoid** data snooping: Always use domain knowledge to create your hypothesis set **before** even looking at the training data
- **Account for** data snooping: If not possible to avoid, look at the data but make sure to account for this

### Pitfalls

- Explicit: Intentionally trying many models on the same dataset, thereby increasing size of hypothesis set
- Implicit
  - Looking at the test data before choosing a model
  - Data leakage during feature engineering, such as normalization

- Adaptive analysis: When working with a public data set, we may already know what models work/don’t work, so the Hypothesis space > the model I formulate

For example:

![image-20240624172742562](./assets/image-20240624172742562.png)

If you look at the data beforehand
$$
\begin{aligned}
H =
&\{
\\
& \quad \{ 1, x_1, x_2, x_1 x_2, x_1^2, x_2^2 \}, \\
& \quad \{ 1, x_1^2, x_2^2 \}, \\
& \quad \{ 1, x_1^2 + x_2^2 \}, \\
& \quad \{ x_1^2, x_2^2 - 0.6 \}
\\
& \}
\end{aligned}
$$

### What to do?

- Formulate the research qn and fix the what model before seeing training data.
- If you intend on data snooping and choose a model based on the data, then you should decide on the set of models you are going to choose from before seeing the data, and account for the data snooping in your analysis by
  - Adjusting the significance level of your hypothesis tests by, for example, using the Bonferroni correction
  - Using a test data set to evaluate the performance of your final estimated model. The test set should be allocated at the beginning and only used at the end. **==Once a data set has been used, it should be treated as contaminated for evaluating test performance==**

### Reporting Guidelines

- Aim for honesty & transparency
- Clearly state research qn, research design, and reasoning behind model choice.
- Clearly state if analysis involves data snooping and how you have accounted for it.
- Report every hypothesis test you have performed relevant to the research question and highlight results that are robust across tests.
- Include a limitations section and point out any limitations and uncertainties in the analysis.

