## Econometric Analysis

Machine Learning $\to$ Statistics $\to$ Econometrics

The focus is on using the predictions of Machine Learning to analyse future causalities. This is more insight-oriented, as identifying patterns without understanding causes and implications is useless.

## Statistical vs Causal

|                                       | Statistical Prediction                                       | Causal Prediction                                            |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\hat y$                              | $E \big [y \vert x = a \big]$                                | $E \big [y \vert \text{do}(x = a) \big]$                     |
| What will be $y$ if I ___<br />$x=a$? | observe                                                      | set                                                          |
| characteristic                        | natural outcome                                              | manual action                                                |
| statement                             | if $x$ is correlated with $y$ without any causal effect on $y$, then we can only observe the correlation without the ability to change $y$ by changing $x$ | if $x$ has a causal effect on $y$, then we can change $x$ and expect it to cause a change in $y$. |
|                                       |                                                              | True understanding enables predictions under a wide range of circumstances, including new hypothetical situations |
|                                       | Unstable                                                     | Stable                                                       |
| Example 1                             | What is the expected health status of someone who has received hospitalization? | What will the health status of a person if they receive hospitalization? |
| Example 2                             | What is the expected sales of a company with a given amount of TV ad spending? | How much will my sales increase if I increase my TV ad spending by a certain amount? |

## see vs do

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

We can now conclude that the barometer reading itself has no causal effect on rainfall. It is infact the atmospheric pressure that has causal effect on the rainfall.

### Hospital

Let’s say there are 2 hospitals

|               | A    | B    |
| ------------- | ---- | ---- |
| Recovery Rate | 0.6  | 0.4  |

It seems like A is better than B. But A may not necessarily be the better hospital for me to go to. What if A is a regular community hospital and B is a speciality hospital for cancer patients. Obviously, A will have a better recovery rate. 
So, statistical prediction (is not wrong) is accurate saying that recovery rate for a patient going to hospital A is 0.6, because we are seeing the patient going there; the patient that goes there is a patient who chose to go there. If they were a serious patient, then they would’ve gone to B.

### Ad-Sales

Sales and Advertisement have a high correlation. But doesn’t necessarily mean that Increasing advertisement will increase sales.
This is because, the observed advertisment could be due to previous sales. So the observed advertisment here is a natural outcome, not an active decision; ie, last year i got high sales profits, so i increased my advertisment budget this year, or i got bad profit so i increased my ad budget to somehow increase the sales. it’s like a reaction, not an active decision.

However, when you actively decide to increase the ad budget, the change in sales depends on the causal effect of ad budget on sales. Why don’t companies do this causal analysis???

1. they don’t know their math and stats :laughing:
2. causal analysis is harder

## Scope

Scope is the set of populations in which a causal effect applies. A causal effect is only meaningful if we can define its scope.

## Populations

|                                           |              Study Population               |               Target Population                |
| ----------------------------------------- | :-----------------------------------------: | :--------------------------------------------: |
|                                           | experiment/observational study is conducted | we’re interested in learning the causal effect |
| Sample                                    |                  specific                   |                    diverse                     |
| Population                                |                  specific                   |                    diverse                     |
| Social, cultural and economic environment |                  specific                   |                    diverse                     |

## Properties of Results

### Validity

|                              |     Internal     |     External      |
| ---------------------------- | :--------------: | :---------------: |
| Results valid for            | study population | Other populations |
| Transportability of results  |        ❌         |         ✅         |

### Transportability

The ability of the result can be generalized/extrapolated correctly from one population to another.

A causal effect learnt from a study is transportable from study population to target population if both are within the scope.

## Importance of Causal Learning

### Russel’s Chicken

This short story shows how pure reliance on past data is bad.

The chicken assumes that whenever the farmer comes, it is to feed it. However, there will one day, the farmer comes to kill it.

Hence, the lack of understanding **why** something happens might be very dangerous.

### 2008 US Financial Crisis

Default prediction was based on the historical data, in which housing prices were always rising

However, this time, the house pricing were going down

### Simpson’s Paradox

This paradox looks at the effectiveness of a drug.

https://youtu.be/ebEkn-BiW5k

[Aggregate Reversal](#Aggregate Reversal)

For example, in this study, the **composition** makes a difference, ie

- in the ‘drug’ group, there are more women than men
- in the ‘no drug’ group, there are more men than women

This disparity will give an incorrect understanding

Moreover, for this particular disease, **women have a lower recovery rate than men.** That should be taken into account as well.

Let’s take another example. Consider a simple example with 5 cats and 5 humans. Let 1 cat and 4 humans be given the drug. Now, the values in the table show the **recovery rate**.

|         |     Drug      |   No Drug    |
| ------- | :-----------: | :----------: |
| Cat     | $1/1 = 100\%$ | $3/4 = 75\%$ |
| Human   | $1/4 = 25\%$  | $0/1 = 0\%$  |
| Overall | $2/5 = 40\%$  | $3/5 = 60\%$ |

If we look at individual groups, cats are better off with drugs, and so are the humans.

However, when we look at overall we can see that the population as a whole is better **without the drugs**.

### US Political Support

Similar to [Simpson’s Paradox](#Simpson’s Paradox)

[Aggregate Reversal](#Aggregate Reversal)

| Level      | Richer you are, more likely to be a __ | Reason                                                       |
| ---------- | -------------------------------------- | ------------------------------------------------------------ |
| Individual | Republican                             | Republican individuals are richer and want lower taxes       |
| State      | Democrat                               | Richer societies are usually morally ‘modern’; Poorer one are usually conservative and religious<br /><br />Democrats have more ‘modern’ policies |

### Survivorship Bias

This wasn’t taught in this course, but i just remembered.

The planes that returned from war had lot of spots with bullet shots.

Some person suggested strengthening only those spots. Initially, that makes sense - these are the areas that got shot so we need to strengthen. But, that is wrong.

Another person said that these are the planes that returned **despite** getting shot at these spots. That means that we have to focus on other places, because the planes that got shot there never returned.

Clearly data can be misleading, without understanding the underlying cause.

## Aggregate Reversal

Any statistical relationship between two variables may be reversed by including additional factors in the analysis

If you just look at statistical data, it might be misleading.

Once we devide the population into sub-population based on categories such as sex, then it becomes clearer. This is because why try understanding the underlying mechanism. This phenomenon is called as **aggregate reversal**.
