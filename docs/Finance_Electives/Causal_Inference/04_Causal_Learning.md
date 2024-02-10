# Causal Learning

## Types

| Causal Effect Learning                                       | Causal Mechanism Learning                                   | Causal Inference Learning                                    |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Does $x$ have a causal effect on y? If yes, how large is the *effect* | If causal effect exists, what is the *mechanism* behind it? | Understand rational decisions that can be taken, built on causal mechanism learning and prior causal inference |
| - What<br />- How much                                       | - Why<br/>- How                                             | - What can we do?                                            |
| - discovering patterns<br />- making predictions             | - understanding                                             | - decison-making                                             |

## Causal Effect Learning

1. Identification: Formulate causal reasoning
2. Reformulate into statistical form
3. Estimation: Use appropriate statistical model

## Causal Mechanism Learning

To understand the true meaning and scope of a causal effect, we need to understand the underlying causal mechanism, based on **prior knowledge** - information and analyses.

This is important to understand

- causal effect - what it means, where it applies
- transportability of results

We have to understand the true mechanism and reason why a treatment affects the outcome. A randomized experiment is not a suitable reason to skip that step.

If the mechanisms in play are globally-applicable, then we can conclude that results of the study can be applied everywhere. Otherwise, we can**not** confidently conclude that.

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

## Limitations of Causal Effect learning

Causal effects do not exist in a vacuum. They are usually the effects of complex and economic processes.

Causal effects are limited to the [scope](#Scope) of the study, ie, population-specific. When we talk about “the causal effect of $x$ on $y$”, it is always wrt to a specific population within a specific social, cultural, and economic enviroment.

This may **not** be accurate because

1. Causal mechanism is different in both populations
   eg: Consider a country where oil prices are determined by market, and another country where prices are determined by govt. The effect of decreasing oil supply on gas price will be different 
2. Distribution of effect modifier $P(s)$ is different
   Eg: Consider 2 countries with the same economic structure, but different population age structures. The effect of raising retirement age will be completely different.

## Causal Inference Models

There are 2 types of models

1. Rubin Model
2. Judea Pearl Model
   The instructor says that this is better, in his opinion

## Identifiability

$\theta(M)$ is if it can be uniquely determined based on observations of $v$.

I didn’t really understand this.

## Probabilistic

Causal effect of a treatment is a probability distribution: it is not the same for every individual.

- Learning the individual-level is nearly impossible
- Learning the pdf of the effect is hard

Hence, we use the Average Treatment Effect