# Econometrics

Seeks to link economic theory to data

> Intermediate between mathematics, statistics, and economics, we find a new discipline which for lack of a better name, may be called econometrics.
>
> Econometrics has as its aim to subject abstract laws of theoretical political economy or ’pure’ economics to experimental and numerical verification, and thus to turn pure economics, as far as possible, into a science in the strict sense of the word.
>
> …
>
> So far, we have been unable to find any better word than "econometrics". We are aware of the fact that in the beginning somebody might misinterpret this word to mean economic statistics only. But ... we believe that it will soon become clear to everybody that the society is interested in economic theory just as much as in anything else.
>
> ~ Ragnar Frisch

## Causal Reasoning

The econometric approach to causal reasoning starts with assuming a “true model”
$$
y = \alpha_0 + \alpha_1 x + u
$$
where

- $x$ is observed
- $u$ is unobserved
- $\alpha_1$ represents the average
  causal effect of $x$ on $y$

To estimate $\beta$, we run a linear regression
$$
y = \beta_0 + \beta_1 x + e
$$
$u$ and $x$ are uncorrelated $\implies$ back-door criterion is satisfied: $x$ is exogenous and OLS estimation produces that $\hat \beta$ that is a unbiased estimate of $\beta$. Else, $x$ is endogenous and $\beta$ is biased estimate of $\beta$

$\beta$ is a statistical parameter; $\alpha$ is a causal parameter

## Limitations

### Statistical vs structural

The econometric approach does not clearly distinguish between causal reasoning and statistical modeling, ie what assumptions are causal and what are statistical
- Just because does $x$ and $y$ are not statistically related, does not mean that $x$ does not cause $y$
- Most econometrics textbooks use one equation to represent both models, confusing what is causal and what is statistical.

Econometric literature states: “When the error term is correlated with the regressor, the estimation result is biased”
- By the “error term”, it is referring to $u$, not $e$
- By “estimation result is biased”, it is saying that $E[\hat \beta] \ne \alpha$
  - ie, $\hat \beta$ is an unbiased estimate of statistical parameter $\beta$, but a biased estimate of the causal parameter $\alpha$

The requirement that $u$ be (linearly) uncorrelated with $x$ is often stated as an essential assumption on the linear regression model itself, under which the OLS estimator is unbiased – again confusing what is causal and what is statistical

Failure to difference b/w statistical & structural equations can lead to confusion.

For eg, the following causal model with the causal effect of $x$ on $y$ is 0, ie no effect
$$
\begin{aligned}
x & ∼ U (0, 1) \\
y & ∼ U (0, 1) \\
u & \leftarrow y−x
\end{aligned}
$$
However, the causal model implies the following statistical equation:
$y = \alpha x + u$ , where $\alpha = 1$ and $u$ is correlated with both $x$ and $y$

Failure to understand this as a statistical rather than structural equation will lead us to wrongly conclude that a regression of y on x will produce biased causal estimates – it won’t. It will only produce a biased estimate of $\alpha$, but $\alpha$ is not the average causal effect of $x$ on $y$

### Identification & Estimation

The econometric approach does not clearly distinguish between nonparametric and parametric identification, and between identification and estimation.

Identification in the econometric approach refers to whether the parameters in a “true model” can be uniquely determined given infinite data on the observed variables. A “true model”, however, already makes parametric assumptions on the underlying causal structure.

Therefore, while causal inference based on causal graphical models recognizes causal effect learning as a two-stage process – identification and estimation – and uses the back-door and front-door criteria to establish clear rules for nonparametric identification, the econometric approach fails to do so. This leads to confusion over how to apply statistical and machine learning models for causal inference.

### Control Variable Selection

The econometric approach does not provide an easily operational way of choosing control variables for identifying a desired causal effect.

If $x$ is endogenous, we can try to find control variables $z$ such that: conditional on $z$, $u$ is no longer correlated with $x$ in the following model
$$
y = \alpha_0 + \alpha_1 x + \alpha_2 z + u
$$
Then, running a regression
$$
y = \beta_0 + \beta_1 x + \alpha_2 z + u
$$
$\hat \beta$ will be an unbiased estimate of $\alpha$

The requirement that $u$ be (linearly) uncorrelated with $x$ conditional on $z$ can be thought of as implied by the back-door criterion. However, unlike the back-door criterion, the econometric approach to causal reasoning does not offer a clear guidance on the choice of $z$, because it is not based on a clear thinking of the underlying causal mechanism (such as a causal diagram representation); clear thinking on causal mechanism forms part of the appeal of structural estimation over the reduced-form approach

The statistics and econometrics literature often state that in order for causal effect to be identifiable by conditioning – there must be no unmeasured confounding (statistics) or selection on observables (econometrics). However, the back-door criterion shows that we do not need to observe and condition on all confounders, only a sufficient set of variables that renders all back-door paths blocked.



There are now more than one “true models”, because for most problems, multiple sets of variables exist that can render all back-door paths blocked.

- The econometrics literature defines omitted variable bias as the bias that arises when a variable is omitted from the “true model.” But with more than one true model – with multiple sets of $z$ that can sufficiently control for confounding – what is an omitted variable?

Meaning of “true model” is now no longer clear. Once we include control variables $z$, “true model” can no longer be interpreted as a structural equation in a causal model

- The back-door criterion makes it clear that z can include not only direct causes of $y$, but also variables that are not causes of $y$. Therefore, “true model” is no longer a structural equation specifying the relation between y and its determinants.
- The meaning of the equation itself has become unclear
- Unclear of its ability to offer meaningful guidance on the choice of $z$
  - The back-door criterion also makes it clear that $z$ should not include descendants of $x$ and should not include colliders that may open a back-door path. The econometric approach offers none of these guidances