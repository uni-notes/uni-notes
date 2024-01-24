# Manipulation of $x$

Being able to manipulate $x$ to see its effect on $y$ is essential to understanding causality. If there is no way to manipulate $x$, then it is difficult to understand causality. 

Morever, according to the instructor, it is pointless to causal inference as if we cannot change it (even theoretically), then we can’t really make better decisions, you know? So many questions we analyze when doing research is basically useless.

For eg, analyzing “what is the causal effect of height on your income”. This is kinda pointless, because it’s not like we can change our height. Atleast “what is the causal effect of democracy on economic growth” is an acceptable analysic, because theoretically we can change the democracy level.

I have an example. Analysing the ‘causal effect of unemployment on economic growth’ is not very useful, because even though we can hypothetically manipulate unemployment indirectly, we can’t exactly control it directly.

## Type of Manipulation

The mechanism with which you ‘do’ $x$ will have different results. Hence, it is important to have a clear mechanism for ‘do’-ing $x$ before starting your analysis.

For eg, for the theoretical democracy example, are you going to forcefully implement a democracy? or will the citizens peacefully request?

# Experimentational Causal Analysis

once the experiment is over, the correlation is mathematically equal to the causation

### Steps

1. manually set $x=1$
1. observe the value of $y$
1. repeat
1. take average value of y

### Disadvantages

1. not always feasible (especially in economics), and it is not possible to perform the experiment
2. everyone is different, the experiment might not give an accurate inference

### Example

RCT (Randomized Control Testing)

- test group is do(x=1) - taking drug
- control group is do(x=0) - not taking drug

# Observational Causal Analysis

We are trying to express causal correlation **in terms** of statistical correlation

1. Reduced Form Econometrics
2. Structural Econometrics

|                | Reduced Form                                                 | Structural                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| models         | mathematical/statistical                                     | economic                                                     |
| accuracy       | low                                                          | high, only if the economic model is good                     |
| time to derive | -                                                            | high                                                         |
| when to use    | prediction for data **within** the range of pre-existing data | prediction for data **outside** the range of pre-existing data |

The shortcoming of reduced form was seen in the 2008 Recession
The prediction model for defaults was only for the case that housing prices go up, as there was data only for that. Hence, the model was not good for when the prices started going down.

# Reduced Form Econometrics

IDK

# Causal Inference in AI

1. how should a robot acquire causual information through interaction with its environment
2. how should a robot receive causal information from humans

According to the lecturer, a lot of modern-day AI is **not** ‘intelligence’. Just because the algorithm can recognize images by trained data is not exactly ‘intelligence’.

> True hallmark of intelligence is the ability to make causal inference, from looking at statistical patterns.

# Limitations of Causal Effect learning

Causal effects do not exist in a vacuum. They are usually the effects of complex and economic processes.

Causal effects are limited to the [scope](#Scope) of the study, ie, population-specific. When we talk about “the causal effect of $x$ on $y$”, it is always wrt to a specific population within a specific social, cultural, and economic enviroment.

This may **not** be accurate because

1. Causal mechanism is different in both populations
   eg: Consider a country where oil prices are determined by market, and another country where prices are determined by govt. The effect of decreasing oil supply on gas price will be different 
2. Distribution of effect modifier $P(s)$ is different
   Eg: Consider 2 countries with the same economic structure, but different population age structures. The effect of raising retirement age will be completely different.

# Causal Inference Models

There are 2 types of models

1. Rubin Model
2. Judea Pearl Model
   The instructor says that this is better, in his opinion

# Identifiability

$\theta(M)$ is if it can be uniquely determined based on observations of $v$.

I didn’t really understand this.
