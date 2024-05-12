# Rubin Model

Also called as Potential Outcomes Framework

We find the ‘treatment effect’ of $x$. This is just a fancy way of saying causal effect

Uses statistical analysis of experiments to model causality

Framework for causal inference that conceptualizes observed data as if they were outcomes of experiments, conducted through
1. actual experiments by researcher(s)
2. observational studies by subjects of the research

## Terms

|                                   | Keyword                                      | Meaning                                                |
| --------------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| $x$                               | Treatment/<br />Intervention/<br />Mediation | input                                                  |
| $y$                               | Outcome                                      | output                                                 |
| $x \perp (y^0, y^1)$              | Exchangeability/Exogenous                    | input is independent                                   |
| $x \perp \!\!\! \perp (y^0, y^1)$ | Conditional exchangeability                  | input is independent only for a certain sub-population |
|                                   | Endogeneous                                  | input is dependent (self-chosen)                       |

## ill-Defined Intervention

When the treatment is not defined specifically, there exists multiple variations of the treatment. Hence, derived effect will not be meaningful, and may be misleading.

### Effect of democracy on economic growth

You need to keep in mind that there are multiple variations of

- democracy - parliamentary, presidential, …
- country becoming democratic - peaceful transition, civil uprising, revolt, …

In this case, the ‘effect of democracy on economic growth’ will not be meaningful, as each of these various treatments will have different outcomes, and cannot be generalized.

### Effect of obesity on health

- What is obesity as a treatment?
- How do we intervene on obesity?
- Multiple channels to becoming obese or un-obese: (lack of) exercise, (un)healthy diet, surgery, ...
- The apparently straightforward comparison of the health outcomes of obese and non-obese individuals masks the true complexity of the interventions “make someone obese” and “make someone non-obese.”

## Potential Outcomes

Consider an input $x_i$ which takes binary values $0/1$. Then, there will be

- 4 potential outcomes
- 2 potential outcomes for each treatment

Suppose the treatment is $x_1$, then

|                             |                          |
| --------------------------- | ------------------------ |
| $x = a$                     | actual treatment         |
| $x \ne a$                   | counterfactual treatment |
| $y^a$                       | realized outcome         |
| $y^{\ne a}$                 | counterfactual outcome   |
| $\{ y_i^a , y_i^{\ne a} \}$ | potential outcomes       |

$$
\begin{aligned}
y &= \sum_{a=1}^A y^a I(x=a)
\\
\text{Binary } x \implies
y &= x y_i^1 + (1-x) y_i^0
\end{aligned}
$$

$x$ has causal effect on $y$ $\iff P(y^0) \ne P(y^1)$, where P is the probability. This is because

- if $x$ has no effect, changing it won’t have any effect on the probability of either outcome, so the probabilities will be equal.
- but if it has effect, then obviously the outcome probabilities will be different

## Effects

Let $\tau = y^1 - y^0$.

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

### ATE

ATE = difference of the mean = mean of the difference
$$
\begin{aligned}
\text{ATE}
=& E(y^1 - y^0) \\
=&
E(y^1 - y^0 | x = 1) \cdot P(x=1) \\
&+
E(y^1 - y^0 | x = 0) \cdot P(x=0) \\
=& \text{ATT}  \cdot P(x=1) + \text{ATU} \cdot P(x=0) \\
=& \int E[y^1 - y^0 \vert s] \cdot p(s) \cdot ds
\end{aligned}
\label{ATE}
$$
This $\eqref{ATE}$ reminds me of the total probability like in Bayes’ conditional probability. But here, we are taking expectation $E$ (mean), because it’ll more accurate than taking one value from the PDF, as $\tau$ is a random variable.

### IDK

|                                                                              | $E[y^1 - y^0]$                                                   | $E[y \vert x=1] - E[y \vert x=0]$|
|---                                                                           | ---                                                              | ---|
|Compares what would happen if the __ sample receives treatment $x=1$ vs $x=0$ | same                                                             | 2 different|
|Provides | Average causal effect | Average difference in outcome b/w sub populations defined by treatment group |
|  | ![image-20240320180040650](./assets/image-20240320180040650.png) | ![image-20240320180026379](./assets/image-20240320180026379.png) |

### IDK

$$
\begin{aligned}

\text{ATT}
&= E(y^1 - y^0 | x = 1) \\
&= \underbrace{E(y^1 | x = 1)}_{E(y | x = 1)} -
\underbrace{E(y^0 | x = 1)}_{\text{Cannot be estimated}}
\\
\text{ATU}
&= E(y^1 - y^0 | x = 0) \\
&= \underbrace{E(y^1 | x = 0)}_{\text{Cannot be estimated}} -
\underbrace{E(y^0 | x = 0)}_{E(y | x = 0)}
\\
%{
%%\text{Similarly,} &\\
%%\text{ATE}
%%&= \underbrace{E(y^1 | x = 0)}_{\text{Cannot be estimated}} -
%%\underbrace{E(y^0 | x = 0)}_{E(y | x = 0)}
%}
\end{aligned}
$$

Solution: Randomized Treatment

## PDF Graph

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

## IDK

| Treatment<br />$x$ | Observed Outcome<br />$y$ | Potential Outcome<br />$y^0$ | Potential Outcome<br />$y^1$ | ITE   |
| ------------------ | ------------------------- | ---------------------------- | ---------------------------- | ----- |
| 0                  | -0.34                     | -0.34                        | 3.46                         | 3.8   |
| 0                  | 1.67                      | 1.67                         | 4.03                         | 2.36  |
| 0                  | -0.77                     | -0.77                        | 3.08                         | 3.85  |
| 0                  | 2.64                      | 2.64                         | 0.90                         | -1.74 |
| 0                  | -0.02                     | -0.02                        | 0.96                         | 0.98  |
| 1                  | 2.31                      | -1.52                        | 2.31                         | 3.83  |
| 1                  | 2.79                      | 1.05                         | 2.79                         | 1.74  |
| 1                  | 1.53                      | -0.13                        | 1.53                         | 1.65  |
| 1                  | 3.61                      | -1.41                        | 3.61                         | 5.02  |
| 1                  | 3.36                      | 0.60                         | 3.36                         | 2.76  |

Here

- Modelling the ITE is correct
- Modelling $y$ vs $x$ is incorrect

## Shortcomings

1. Since it is more experiment-oriented, it is hard to analyze continuous treatment. It is only feasible to do binary $0/1$ treatment.
2. Cannot learn individual treatment effects, since counterfactual outcomes are not observed. This is the **fundamental problem of causal inference**
3. We can only learn causal effects at ~~population~~ sample level
   1. Therefore, when learning a causal effect, we should always be clear
      about the ~~population~~ sample on which it is defined
4. According to Rubin, causal inference is a ‘missing data’ problem, but that’s just like every other statistical predictive model
5. It does **not** model choice as assignment of unit’s ability and eligibility for treatment; it models model choice as assignment to a treatment
