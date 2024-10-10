# Hypothesis Testing

Hypothesis testing involves testing the following question

> Is the estimated value sufficiently close to stated value?

## Hypothesis

- Simple Hypothesis
    - Single Tailed
    - $\beta_2 > 0.3$
- Composite Hypothesis
    - 2 Tailed
    - Bi-Directional
    - Useful when not sure about the direction
    - $\beta_2 \ne 0.3$

### Null Hypothesis $(H_0)$

Your initial statement

### Alternative Hypothesis $(H_1)$

[Usually] complement of your initial statement

Also called as **Maintained Hypothesis**. It acts as the fallback in case null hypothesis is proven to be false.

Then the value of $y$ is taken to be the value obtained from the sample

### Number of Hypotheses

If $n =$ no of independent variables, the number of hypothesis is $2n + 1$

$+1$ is due to intercept(constant)

## Steps

1. Formula hypotheses

2. Determine if one/two tailed test

3. Construct a $100(1-\alpha) \ \%$ confidence interval for $\beta_2$

4. Determine critical values

5. Determine rules to accept/reject null hypothesis

6. Compare estimate-value with critical region

7. Conclusion

     - If it lies within critical region, accept null hypothesis
     - If it lies outside critical region, reject null hypothesis
     - accept alternate hypothesis
     - $\beta_2$ will take the sample value

## Confidence Interval

$$
\begin{aligned}
(1-\alpha)
&= P(- t_{\alpha/2} \le
\textcolor{hotpink}{t}
\le +t_{\alpha/2}) \\
\textcolor{hotpink}{t}
&= \frac{\hat \beta_2 - \beta_2}{\sigma(\hat \beta_2)}
\end{aligned}
$$

$$
(1-\alpha) =
P(\hat \beta_2  t)
$$

- $\alpha =$ level of significance
- $(1-\alpha) =$ Confidence coefficient

Construct the confidence interval for $t$ distribution, with $(n-2)$ degrees of freedom.
This is because we have 2 unknowns.

![confidence](assets/confidence.svg){ loading=lazy }

## Level of Significance

Tolerance level for error

This is

- probability of committing type 1 error
- probability of rejecting null hypothesis, and then getting sample value as the actual value just by chance

| Field           | Conventional $\alpha$ | Conventional $(1 - \alpha) \%$ |
| --------------- | --------------------: | -----------------------------: |
| Pure Sciences   |                  0.01 |                            99% |
| Social Sciences |                  0.05 |                            95% |
| Psychology      |                  0.10 |                            90% |

## Normal Distribution

- $95 \%$ values lies within 1 standard deviation on each side from the center
- $2.5 \%$ values lies outside 1 standard deviation on left side
- $2.5 \%$ values lies outside 1 standard deviation on right side

## Errors

|              | Type 1                                                       | Type 2                              |
| ------------ | ------------------------------------------------------------ | ----------------------------------- |
| Error of     | Rejecting correct null hypothesis                            | Accepting incorrect null hypothesis |
| Meaning      | False Negative                                               | False Positive                      |
| Measured by  | $\alpha$<br />([Level of Significance](#Level-of-Significance)) |                                     |
| Happens when | Sample is not a good representation of population            |                                     |

## Statistical Equivalence

0.5 can be statistically = 0, or not; depends on the context

$$
\begin{aligned}
P(\text{rejecting } H_0)
&\propto |t| \\
&\propto \text{Deviation of sample value from true value}
\end{aligned}
$$

$$
t = 0 \implies \hat \beta_2 = \beta_2
$$

## p-Value

Observed level of significance

==$\text{p value} \le \alpha \implies$ Reject null hypothesis==

## $t=2$ Rule

For degree of freedom $\ge 20$

### 2 Tailed
$H_0: \beta_2 = 0, H_0: \beta_2 \ne 0$ 

If $|t| > 2 \implies p \le 0.05 \implies$ reject $H_0$

### 1 Tailed

$H_0: \beta_2 = 0, H_0: \beta_2 > 0$ or $H_0: \beta_2 = 0, H_0: \beta_2 < 0$ 

If $|t| > 1.73 \implies p \le 0.05 \implies$ reject $H_0$

## Why $t$ distribution?

$t$ distribution is a variant of $z$ distribution

- For small samples, we use $t$ dist
- For large sample, we use $z$ dist
