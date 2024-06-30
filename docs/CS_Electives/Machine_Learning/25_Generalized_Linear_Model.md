# Generalized Linear Model

Why? For non-normal distribution, OLS $\ne$ MLE

Steps

1. Let $y$ have a probability distributions as long as it is from the exponential family

   - Included

     - Normal, log-normal, exponential, gamma, chi-squared, beta, Bernoulli, poisson, binomial, etc

   - Not included:

     - Student’s $t$ due to heavy tails

     - Mixed distributions (with different location/scale parameters)

2. Allow for any transformation (link function) of $y$, such that transformation is monodic and differentiable

3. Write linear parameters

4. Derive MLE

| Distribution             | Typical Uses                                       | Link Name             | Link Function<br />$g(y)$                     |
| ------------------------ | -------------------------------------------------- | --------------------- | --------------------------------------------- |
| Bernoulli/<br />Binomial | Outcome of single yes/no occurence                 | Logit<br />(Logistic) | $\ln \left \vert \dfrac{y}{1-y} \right \vert$ |
| Exponential/<br />Gamma  | Exponential response data<br />Scale parameters    | Inverse               | $1/y$                                         |
| Normal/<br />Gaussian    | Linear response data                               | Identity              | $y$                                           |
| Inverse Gaussian         |                                                    |                       |                                               |
| Poisson                  | Count of occurrences in fixed amount of time/space | Log                   | $\ln \vert y \vert$                           |
| Quasi                    | Normal with constant variance                      |                       |                                               |
| Quasi-binomial           | Binomial with constant variance                    |                       |                                               |
| Quasi-poisson            | Poisson with constant variance                     |                       |                                               |

