# Rubin Model

Potential Outcomes Framework

We find the ‘treatment effect’ of $x$. This is just a fancy way of saying causal effect

> framework for causal inference that conceptualizes observed data as if they were outcomes of experiments, conducted through
>
> 1. actual experiments by researcher(s)
> 2. observational studies by subjectso of the research

# Terms

|                                   | Keyword                     | Meaning                                                |
| --------------------------------- | --------------------------- | ------------------------------------------------------ |
| $x$                               | treatment/intervention      | input                                                  |
| $y$                               | outcome                     | output                                                 |
| $x \perp (y^0, y^1)$              | Exchangeability/Exogenous   | input is independent                                   |
| $x \perp \!\!\! \perp (y^0, y^1)$ | Conditional exchangeability | input is independent only for a certain sub-population |
|                                   | Endogeneous                 | input is dependent (self-chosen)                       |

# ill-Defined Intervention

When the treatment is not defined specifically, there exists multiple variations of the treatment. Hence, derived effect will not be meaningful, and may be misleading.

For example, when studying the ‘effect of democracy on economic growth’, you need to keep in mind that there are multiple variations of

- democracy - parliamentary, presidential, …
- country becoming democratic - peaceful transition, civil uprising, revolt, …

In this case, the ‘effect of democracy on economic growth’ will not be meaningful, as each of these various treatments will have different outcomes, and cannot be generalized.

# Potential Outcomes

Consider an input $x_i$ which takes binary values $0/1$. Then, there will be

- 4 potential outcomes
- 2 potential outcomes for each treatment

Suppose the treatment is $x_1$, then

|                     |                          |
| ------------------- | ------------------------ |
| $x_0$               | counterfactual treatment |
| $x_1$               | actual treatment         |
| $y_1^0$             | counterfactual outcome   |
| $y_1^0$             | realized outcome         |
| $y_1^0$ and $y_1^1$ | potential outcomes       |

$$
y = x y_i^1 + (1-x) y_i^0
$$

$x$ has causal effect on $y$ $\iff P(y_i^0) \ne P(y_i^1)$, where P is the probability. This is because

- if $x$ has no effect, changing it won’t have any effect on the probability of either outcome, so the probabilities will be equal.
- but if it has effect, then obviously the outcome probabilities will be different

# Average

Let $\tau_i = y_i^1 - y_i^0$.

$\tau$ will have a distribution because it is a random variable ($\tau_1, \tau_2, \dots, \tau_i$)

|                  | Term | Meaning                                                      |
| ---------------- | ---- | ------------------------------------------------------------ |
| $\tau_i$         | ITE  | Individual Treament Effect<br />it is never observed, because we only observe $y_i^0$ or $y_i^1$ |
| $E[\tau]$        | ATE  | Average Treatment Effect                                     |
| $E[\tau |x = 1]$ | ATT  | Average Treatment effect on Treated                          |
| $E[\tau|x = 0]$  | ATU  | Average Treatment effect on Untreated                        |
| $E(y^1)$         |      | Expectation of outcome 1 of the entire population (hypothetical, counterfactual) |
| $E(y^1 | x = 1)$ |      | Expectation of outcome 1 of the **treated sample**           |

Why are there 3 different average variables?
The people in each group is different. So, $\tau$ for the entire group, treated and untreated groups are different, due to ‘selection effect’. This is like people who go to uni vs don’t.
$$
\begin{aligned}
y &= y^0 \cdot I(x=0) \times y^1 \cdot I(x=1)\\
&\text{where $I$ means if}\\

\implies y &= f(x, y^0, y^1)
\end{aligned}

\label{y}
$$

## ATE

ATE = difference of the mean = mean of the difference
$$
\begin{aligned}
 &\text{ATE} \\
=& \text{ATT - ATU} \\
=& E(y^1 - y^0) \\
=&
\begin{split}
E(y^1 - y^0 | x = 1) \times P(x=1) \\
+
E(y^1 - y^0 | x = 0) \times P(x=0)
\end{split}
\end{aligned}

\label{ATE}
$$
This $\eqref{ATE}$ reminds me of the total probability like in Bayes’ conditional probability. But here, we are taking expectation $E$ (mean), because it’ll more accurate than taking one value from the PDF, as $\tau$ is a random variable.

## ATT/ATE

$$
\begin{aligned}

\text{ATT}
&= E(y^1 - y^0 | x = 1) \\
&= \underbrace{E(y^1 | x = 1)}_{E(y | x = 1)} -
\underbrace{E(y^0 | x = 1)}_{\text{Cannot be estimated}}
\\

\text{Similarly,} &\\
\text{ATE}
&= \underbrace{E(y^1 | x = 0)}_{\text{Cannot be estimated}} -
\underbrace{E(y^0 | x = 0)}_{E(y | x = 0)}

\end{aligned}
$$

# PDF Graph

![potentialOutcomesDistribution](assets/potentialOutcomesDistribution.png)

$$
\begin{aligned}
\text{ATE}(x) &=
\int \tau(x, y) P(y) \ dy \\
&= \frac{
	dE[y| \text{ do}(x)]
}{dx} \\

T(x, y) &=
\frac{
	\partial P(y | \text{do}(x))
}{
	\partial x
} \\
\end{aligned}
$$


We could also interpret this entire distribution as a 3 variable joint PDF of the form $P(x, y^0, y^1)$

# Shortcomings

1. Since counterfactual outcomes are not observed, we are not able to learn individual treatment effects. This is the **fundamental problem of causal inference**
2. We can only learn causal effects at population level
3. Since it is more experiment-oriented, it is hard to analyze continuous treament. It is only feasible to do binary $0/1$ treatment.
4. According to Rubin, causal inference is a ‘missing data’ problem, but that’s just like every other statistical predictive model