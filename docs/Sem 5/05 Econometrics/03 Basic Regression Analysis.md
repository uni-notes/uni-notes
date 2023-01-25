## Regression

Examine relationship between different variables

Dependence of one variable on another variable

Identify PRF, using SRF

### Assumptions

- Dependent var $\to$ Random
  Variable whose distribution changes for different variables
- Independent $\to$ Non-Random

### Purpose

Derive a function that traces through the conditional means of $y$ corresponding to different values of $x$

Expected value = mean = average
Same meaning

$$
E(y|x_i)
= \beta_0 + \beta_1 x_i

$$

## Population

not necessarily humans

It refers to any set of data (universal); different from sample (will be covered later).

## PRF

Population Regression Function

Also called as Conditional Expectation Function(CEF)

It is theoretical; we almost never have access to this

It is always linear wrt hyper-parameters, but may/may not be linear wrt variables

## Linearity

|                           | Linear wrt variables        | Non-Linear wrt variables        |
| ------------------------- | --------------------------- | ------------------------------- |
| Linear wrt parameters     | $\beta_0 + \beta_1 x_1$     | $\beta_0 + \beta_2 {x_i}^2$     |
| Non-Linear wrt parameters | $\beta_0 + {\beta_1}^2 x_1$ | $\beta_0 + {\beta_1}^2 {x_1}^2$ |

### Transformation

$$
y_t = e^\alpha x^\beta_t e^{u_t} \iff \ln y_t = \alpha + \beta \ln x_t + u_t

$$

One more thing in slide

Some models cannot be changed;  they are intrinsically non-linear

$$
y_t = \alpha + x^{\color{orange} \beta}_t + u_t

$$

## Stochastic Specification of PRF

$$
\begin{align}
y_i &= E(y|x_i) + u_i\\&\updownarrow\\u_i &= y_i - E(y|x_i)
\end{align}
$$

### Components

- Systematic/Deterministic/Common/Explained component
- Non-Systematic/Random/Disturbance component
  Incorporates
    - effect of all omitted variables
    - random effects
    - effect of measurement error

### Equivalency with PRF

Stochastic Specification is equal to PRF, as long as ==$E(u_i|x_i) = 0$==; this does **not** mean that $u_i = 0 , \forall i$

Why? This is because only if it is so, the line passes through the expectations of $y$ for different values of $x$. It is mathematically possible only if so. (Draw graph and see)
$$
E(y_i | x_i) = E(y|x_i) + E(u_i|x_i)
$$

### Why do we need Stochastic Specification?

- Vagueness of theory
    - Social Sciences has no definite theory for any event
- Randomness in human behavior
- Incorporates effect of missing data
    - Wealth data is not as easy to get as income data
- More appropriate for inexact relatioships
- Captures effect of omitted variables
    - Some variables are not as important
- Captures effect of poor proxy variables
- Principle of Parsimony
  We usually try to limit to simple models
- Incorrect functional form
    - Unknown theory
    - Linear/Non-Linear function
- Incorporates measurement errors

## Proxy Variable

A variable that is closely-associated with the variable we want to use.

We use proxy variables, when the main variable is not available

eg:

- Age and Experience
- CPI and Inflation

## ARCH

Auto Regressive Conditional Heteroscedacity

## Types of Relationships

|                 | Statistical/Schochastic | Deterministic |
| --------------- | ----------------------- | ------------- |
| Independent var | Non-Random              | Non-Random    |
| Dependent var   | Random                  | Non-Random    |
| eg              | Predicting Crop Yield   | Ohm’s Law     |

## Terms

| $y$                                                          | $x$                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dependent<br />Explained<br />Predictand<br />Regressand<br />Response<br />Exogeneous | Independent<br />Explanatory<br />Predictor<br />Regressor<br />Stimulus<br />Endogeneous |

## Capital Flight

Capital moves from one country to another

## Regression $\ne$ Causation

Does not help understand the direction of causality

We need to use domain knowledge, and impose restriction that $x$ causes $y$

## Regression vs Correlation

|            | Regression                                                | Correlation                                          |
| ---------- | --------------------------------------------------------- | ---------------------------------------------------- |
| Understand | Exact relationship                                        | Degree of **linear** association between 2 variables |
| Assumption | One Dependent variable<br />One/more independent variable | Both $x$ and $y$ are random                          |

## Exogeneous vs Endogeneous

Exogeneous vs endogeneous depends on what you assume to be the system

|                 | Exogeneous | Endogeneous |
| --------------- | ---------- | ----------- |
| In our control? | ❌          | ✅           |

- Exo = out
- Endo = in

## Basic Concepts of Regression

1. Derive Conditional values of $y$ wrt $x$
2. Calculate Conditional probabilities of $y$ for different values of $x$
3. Calculate conditional mean
4. Calculate the weighted average using probality of occurance
     - This is different mean from arithmetic mean(simple average)

The expected value of unconditional random variable $y$ is ???

## Variability of $y$

The variation of $y$ for different values of $x$

Higher variability is preferred
