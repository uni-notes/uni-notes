# Introduction

## Econometric Analysis

Machine Learning $\to$ Statistics $\to$ Econometrics

The focus is on using the predictions of Machine Learning to analyse future causalities. This is more insight-oriented, as identifying patterns without understanding causes and implications is useless.

Causation depends on
- Correlation: $x$ and $y$ are statistically-related
- Time order: $x \to y; \text{not } y \to x$
- Non-spuriousness

## Statistical vs Causal

|                                       | Statistical Prediction                                                                                                                                     | Causal Prediction                                                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| $\hat y$                              | $E \big [y \vert x = a \big]$                                                                                                                              | $E \big [y \vert \text{do}(x = a) \big]$                                                                          |
| What will be $y$ if I ___<br />$x=a$? | observe                                                                                                                                                    | set                                                                                                               |
| characteristic                        | natural outcome                                                                                                                                            | manual action                                                                                                     |
| statement                             | if $x$ is correlated with $y$ without any causal effect on $y$, then we can only observe the correlation without the ability to change $y$ by changing $x$ | if $x$ has a causal effect on $y$, then we can change $x$ and expect it to cause a change in $y$.                 |
|                                       |                                                                                                                                                            | True understanding enables predictions under a wide range of circumstances, including new hypothetical situations |
|                                       | Unstable                                                                                                                                                   | Stable                                                                                                            |
| Example 1                             | What is the expected health status of someone who has received hospitalization?                                                                            | What will the health status of a person if they receive hospitalization?                                          |
| Example 2                             | What is the expected sales of a company with a given amount of TV ad spending?                                                                             | How much will my sales increase if I increase my TV ad spending by a certain amount?                              |

## `see` vs `do`

- $\text{see}(x= 1)$ means that you **observe** $x$ as 1
- $\text{do}(x= 1)$ means that you manually **perform some action** to set $x$ as 1

If there is no way to manipulate a variable (for eg, $\text{do}(x= 1)$ is not possible), then it is hard to define what its causal effect means

- A thought experiment that is often used to determine whether a variable x is manipulable in principle is to imagine a hypothetical experiment that assigns different values to x
- Research questions that cannot be answered by any experiment are FUQ’d: Fundamentally Unidentified Questions

## Correlation $\centernot \implies$ Causation

Correlation doesn’t always imply causation. correlation is useful for prediction, but not for understanding exactly why.

For eg, labor wage and their years of education has a strong correlation, but the reason for that could be

- education actually helps
- education just acts as the signal/proof for the employees that you possess knowledge
- or, it could just be that well-off people get better jobs, and coincidentally, they are getting more educated

### Rain

- Barometer reading itself has no causal effect on rainfall
- Atmospheric pressure has causal effect on the rainfall

### Hospital

Let’s say there are 2 hospitals

|               | A   | B   |
| ------------- | --- | --- |
| Recovery Rate | 0.6 | 0.4 |

- It seems like A is better than B. But A may not necessarily be the better hospital for me to go to
- What if A is a regular community hospital and B is a speciality hospital for cancer patients. Obviously, A will have a better recovery rate. 
- So, statistical prediction (is not wrong) is accurate in saying that recovery rate for a patient going to hospital A is 0.6, because we are seeing the patient going there; the patient that goes there is a patient who chose to go there. If they were a serious patient, then they would’ve gone to B.
- But we cannot conclude from this data on how good the hospitals would be for a random person

### Ad-Sales

- Sales and Advertisement have a high correlation. But doesn’t necessarily mean that increasing advertisement will increase sales.
- This is because, the observed advertisement could be due to previous sales. So the observed advertisement here is a natural outcome, not an active decision; ie, last year i got high sales profits, so i increased my advertisement budget this year, or i got bad profit so i increased my ad budget to somehow increase the sales. it’s like a reaction, not an active decision.
- However, when you actively decide to increase the ad budget, the change in sales depends on the causal effect of ad budget on sales

Why don’t companies do this causal analysis???

1. they don’t know their math and stats :laughing:
2. causal analysis is harder

## Scope

Scope is the set of populations in which a causal effect applies. A causal effect is only meaningful if we can define its scope.

## Populations

|                                           |              Study Population               |               Target Population                |
| ----------------------------------------- | :-----------------------------------------: | :--------------------------------------------: |
| Population where                          | experiment/observational study is conducted | we’re interested in learning the causal effect |
| Sample                                    |                  specific                   |                    diverse                     |
| Population                                |                  specific                   |                    diverse                     |
| Social, cultural and economic environment |                  specific                   |                    diverse                     |

The causal effect of $x$ on $y$ can differ in 2 populations because:

1. Causal mechanism is different in both populations

   eg: Consider a country where oil prices are determined by market, and another country where prices are determined by govt. The effect of decreasing oil supply on gas price will be different 

2. Distribution of effect modifier $P(s)$ is different

   Eg: Consider 2 countries with the same economic structure, but different population age structures. The effect of raising retirement age will be completely different.

## Properties of Results

### Validity

| Validity               | Meaning                                                       | Results valid for | Transportability of results | Threats                                                                                                                                                                                   | Solution                                                                               |
| ---------------------- | ------------------------------------------------------------- | :---------------- | :-------------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Construct              | Is your outcome measure the right one to measure the outcome? | Nowhere           |              ❌              | Wrong KPI<br><br>Example:<br>Measuring student commitment with grades is not the best                                                                                                     | Correct KPI                                                                            |
| Statistical Conclusion | Are your statistics correct?                                  | Nowhere           |              ❌              | Low statistical power                                                                                                                                                                     | More observations                                                                      |
|                        |                                                               |                   |                             | Violating assumptions of statistical tests                                                                                                                                                |                                                                                        |
|                        |                                                               |                   |                             | Fishing & p-hacking                                                                                                                                                                       | Don't run different DAGs on all the data; follow good train-test split                 |
|                        |                                                               |                   |                             | Spurious statistical significance                                                                                                                                                         |                                                                                        |
| Internal               |                                                               | Study population  |              ❌              | Calibration: Measurement error                                                                                                                                                            |                                                                                        |
|                        |                                                               |                   |                             | Calibration: Time frame<br><br>If study is too short, effect may not be detectable yet<br><br>If study is too long, attrition may occur                                                   | Choose optimal time frame using domain knowledge                                       |
|                        |                                                               |                   |                             | Contamination: Hawthorne effect<br><br>Observing people makes them behave differently                                                                                                     | Use completely-unobserved control groups                                               |
|                        |                                                               |                   |                             | Contamination: John Henry effect<br><br>Control group works hard to prove they're as good as treatment group<br><br>Usually this happens because control group knows about the experiment | Keep treatment and control groups separate and unaware of each other                   |
|                        |                                                               |                   |                             | Contamination: Spillover effect<br><br>Control groups naturally pick up what treatment group is getting<br><br>Causes: Externalities, social interaction, equilibrium effects             | Keep treatment and control groups separate and unaware of each other                   |
|                        |                                                               |                   |                             | Contamination: Intervening events<br><br>Something happens that affects one of the groups and not the other<br><br>eg: Natural disasters, random events                                   | No fix :/                                                                              |
|                        |                                                               |                   |                             | Omitted variable bias: Self-Selection bias (who opts-in), Attrition (who opts-out), Time selection bias                                                                                   | Randomize and/or inspect characteristics of who joins, who stays, who leaves, and when |
|                        |                                                               |                   |                             | Trends<br>eg: Child growth                                                                                                                                                                | Use control group to remove trend                                                      |
|                        |                                                               |                   |                             | Structural breaks <br>eg: Recessions, cultural shifts                                                                                                                                     | Use control group to remove trend                                                      |
|                        |                                                               |                   |                             | Seasonality                                                                                                                                                                               | Compare observations from same season                                                  |
|                        |                                                               |                   |                             | Testing: Repeated exposure to questions/tasks will make people improve naturally                                                                                                          | Change tests<br>Use control group that receives the test                               |
|                        |                                                               |                   |                             | Regression to the mean<br><br>super-high/super-low performers are systematically-different from the rest of the sample                                                                    | Don't select super-high/super-low performers                                           |
| External               | Generalizability                                              | Other populations |              ✅              | Study volunteers may be W.E.I.R.D.                                                                                                                                                        |                                                                                        |
|                        |                                                               |                   |                             | Not everyone takes surveys/calls<br><br>People who take surveys are systematically different from general population                                                                      |                                                                                        |
|                        |                                                               |                   |                             | Different settings and circumstances                                                                                                                                                      |                                                                                        |

### Transportability

The ability of the result can be generalized/extrapolated correctly from one population to another.

A causal effect learnt from a study is transportable from study population to target population if both are within the scope.

