# Causal Learning

## Types

| Causal Effect Learning                                       | Causal Mechanism Learning                                   | Causal Inference Learning                                    |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Does $x$ have a causal effect on y? If yes, how large is the *effect* | If causal effect exists, what is the *mechanism* behind it? | Understand rational decisions that can be taken, built on causal mechanism learning and prior causal inference |
| - What<br />- How much                                       | - Why<br/>- How                                             | - What can we do?                                            |
| - discovering patterns<br />- making predictions             | - understanding                                             | - decison-making                                             |
| “Effects of causes”                                          | “Causes of effects”                                         |                                                              |

## Manipulation of $x$

Being able to manipulate $x$ to see its effect on $y$ is essential to understanding causality. If there is no way to manipulate $x$, then it is difficult to understand causality. 

Morever, according to the instructor, it is pointless to causal inference as if we cannot change it (even theoretically), then we can’t really make better decisions, you know? So many questions we analyze when doing research is basically useless.

For eg, analyzing “what is the causal effect of height on your income”. This is kinda pointless, because it’s not like we can change our height. Atleast “what is the causal effect of democracy on economic growth” is an acceptable analysic, because theoretically we can change the democracy level.

I have an example. Analysing the ‘causal effect of unemployment on economic growth’ is not very useful, because even though we can hypothetically manipulate unemployment indirectly, we can’t exactly control it directly.

### Type of Manipulation

The mechanism with which you ‘do’ $x$ will have different results. Hence, it is important to have a clear mechanism for ‘do’-ing $x$ before starting your analysis.

For eg, for the theoretical democracy example, are you going to forcefully implement a democracy? or will the citizens peacefully request?

## Experimentational Causal Analysis

once the experiment is over, the correlation is mathematically equal to the causation

#### Steps

1. manually set $x=1$
1. observe the value of $y$
1. repeat
1. take average value of y

#### Disadvantages

1. not always feasible (especially in economics), and it is not possible to perform the experiment
2. everyone is different, the experiment might not give an accurate inference

#### Example

RCT (Randomized Control Testing)

- test group is do(x=1) - taking drug
- control group is do(x=0) - not taking drug

## Causal Inference in AI

1. how should a robot acquire causual information through interaction with its environment
2. how should a robot receive causal information from humans

According to the lecturer, a lot of modern-day AI is **not** ‘intelligence’. Just because the algorithm can recognize images by trained data is not exactly ‘intelligence’.

> True hallmark of intelligence is the ability to make causal inference, from looking at statistical patterns.

## Causal Inference Models

There are 2 types of models

1. Rubin Model
2. Judea Pearl Model
   The instructor says that this is better, in his opinion

## Identifiability

$\theta(M)$ is if it can be uniquely determined based on observations of $v$.

I didn’t really understand this.

## IDK

Requires prior knowledge regarding the data-generating causal mechanism.

Such knowledge can only exist as a result of previously-observed information and conducted studies.

Hence, causal inference builds on past causal inference

## Source of Associations

Reasons why $x$ and $y$ can be associated

- $x$ causes $y$ directly
- $x$ causes $y$ indirectly
- $x$ and $y$ have common cause(s)
- Analysis is conditioned on their common descendant(s)

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

## Sampling Bias

### Sample-selection bias

Type of sampling bias that arises when we make inference about a larger population from a sample that is drawn from a distinct subpopulation

Sample-selection bias can be thought of as a missing
data problem, where data are NMAR (not missing at random)

### Survivorship/Survival Bias

Special type of sample-selection bias

#### Mutual Fund Performance

Suppose we are interested in how the size of assets under management affects a fund’s performance. If we simply look at the relationship between fund size and returns among existing funds, however, there will be what is referred to as a survival bias: we do not observe funds that have closed due to bad performance.

So if fund size negatively affects performance, we may end up under-estimating the magnitude of the effect.

#### Planes in war

The planes that returned from war had lot of spots with bullet shots.

Some person suggested strengthening only those spots. Initially, that makes sense - these are the areas that got shot so we need to strengthen. But, that is wrong.

Another person said that these are the planes that returned **despite** getting shot at these spots. That means that we have to focus on other places, because the planes that got shot there never returned.

Clearly data can be misleading, without understanding the underlying cause

### Wages

![image-20240418221335444](./assets/image-20240418221335444.png)

### Credit card default

![image-20240418221506468](./assets/image-20240418221506468.png)

We cannot use the relationship between Income, balance, and default status for credit card holders
to predict default rate **for a random credit card applicant**, since these people part of the available data have been filtered already as potentially good credit card users

Hence, we can only use it predict default for a random person already having a credit card

This is a case of censoring

#### Success Stories

Advice by someone successful

### 

## Aggregate Reversal

Any statistical relationship between two variables may be reversed by including additional factors in the analysis

If you just look at statistical data, it might be misleading.

Once we devide the population into sub-population based on categories such as sex, then it becomes clearer. This is because why try understanding the underlying mechanism. This phenomenon is called as **aggregate reversal**.
