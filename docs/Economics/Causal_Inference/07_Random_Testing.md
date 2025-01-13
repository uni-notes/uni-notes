# Randomized Tests

## Randomization

- Ensures that correlation = causation
- Eliminates self-selection and trend effects

Helps estimate counterfactual outcome by ensuring
- $y^0, y^1 \perp \!\!\! \perp x$: independence/exchangeability of $y^0, y_1$ wrt $x$
  - Treatment $x$ is exogenous
- Similarity of population and treated sample

$$
\text{ATE = ATT = ATU} \notag
$$

### Independence

If you randomly assign $x$ to people, then the $x$ will be independent from everything. $x$ will therefore be independent from $y_0$ and $y_1$.

If $P(y|x=0) = P(y|x=1) = P(y)$, then $x$ and $y$ are independent. This is because, whatever the value of $x$ we put, the probability of $y$ did not get changed.

So conversely, if we are able to make $x$ and $y^0$ independent, then we can take
$$
E(y^0 | x = 1) =
E(y^0 | x = 0) =
E(y | x = 0)
$$

$$
\begin{aligned}
E(y^1 - y^0) &=
E(y^1 | x = 1) - E(y^0 | x = 0) \\
&= E(y | x = 1) - E(y | x = 0)
\end{aligned}
$$

### Similarity

1. Randomly assign individuals to treatment and control group
2. Ensure that demographics and confounders are balanced in each group
3. Do **not** control any other variable after this for running regressions, as you will open causal paths

### Derivations from Randomized Tests

| Property                         | Meaning                                            |
| -------------------------------- | -------------------------------------------------- |
| Association (Correlation)        | $P(y|x=a) \ne P(y|x=b)$                            |
| Causation                        | $P(y^a) \ne P(y^b)$                                |
| Correlation $\implies$ Causation | Random assignment of $x \implies (y^a) = P(y|x=a)$ |

## Effect of Randomization on deriving ATE

### Without Randomization

There will be statistical correlation without causal correlation (to be avoided). This is due to [Self-Selection Effect](#Self-Selection Effect). So the selection will be biased.

$$
\begin{aligned}
& E(y | x = 1) - E(y | x = 0)  \\
=& 
\underbrace{
	E( y|x = 1) \textcolor{orange} {- E(y^0|x=1)}
}_\text{ATT}
\\
&+
\underbrace{
\textcolor{orange}{ E(y^0|x=1) } - E(y | x = 0)
}_{\text{Self-Selection Bias } \ne \text{ATU} }
\\

\ne & E(y' - y^0) \\
\ne & \text{ATE}
\end{aligned}
$$

### With Randomization

$$
\begin{aligned}
A \perp \!\!\! \perp B &\iff P(A) = P(A \vert B=0) = P(A \vert B=1) \\
\\
\implies
y^0 \perp \!\!\! \perp x &\iff P(y^0) = P(y^0 \vert x=0) = P(y^0 \vert x=1) = P(y \vert x=0) \\
\implies
y^1 \perp \!\!\! \perp x &\iff P(y^1) = P(y^1 \vert x=0) = P(y^1 \vert x=1) = P(y \vert x=1)
\end{aligned}
$$

$$
E[y^1 - y^0] = E[y \vert x=1] - E[y \vert x=0]
$$

$$
\begin{aligned}
&\text{ATE} = \text{ATT} = \text{ATE} \\
&= E[y \vert x=1] - E[y \vert x=0]
\end{aligned}
$$

This only applies since main is a linear operator. This does **not** apply for non-linear operators: median, variance, percentiles

## Conditional Randomized Experiments

If one or more external parameters affect the causal effect of the treatment on the outcomes, then we have to do different randomized conditions.

By independent testing at different conditions, we can keep the effect of the external parameter as a constant, and we will get the true causal effect of the treatment.

It leads to **conditional exchangeability**, for the particular sub-population
$$
x \perp \!\!\! \perp
(y^1, \dots, y^A) | s
$$
where $s$ are the fixed [Effect Modifiers](## Effect Modifiers)
$$
\begin{aligned}
E[y^a]
&= \sum_{j=1}^S E[y^a \vert s=j] \cdot p(s=j) \\
&= \sum_{j=1}^S E[y \vert x=a, s=j] \cdot p(s=j)
\end{aligned}
$$
In experimental design, effect modifiers $s$ are called the nuisance factors that experimenter controls when performing the RCT. Nuisance factors are vars that can affect $y$ either directly/indirectly, but is not of primary interest to the experimenter.

## Self-Selection Effect/Bias

In economics, we assume that everyone

- is rational
- makes decisions/selections to maximize self-interests

When individuals choose their own treatments, those who choose to receive a treatment may be systematically different than those who choose not to, leading to a correlation between treatment and outcome that is not due to direct causation

## Importance of Control Group

We always need to have control group $x=0$, to effectively quantify the causal effect.

Let’s say $x$ is non-binary, for eg: $x=$ Sunlight

- 0: Rainy
- 0.5: Cloudy
- 1: Sunny

| $E(y \vert \text{do}(x=0))$ | $E(y \vert \text{do}(x=0.5))$ | $E(y \vert \text{do}(x=1))$ | Conclusion:<br />Changing from 0.5 to 1 is significant |
| --------------------------- | ----------------------------- | --------------------------- | ------------------------------------------------------ |
| 0                           | 100                           | 101                         | ❌                                                      |
| 0                           | 0                             | 101                         | ✅                                                      |
| 100                         | 100                           | 101                         | ✅                                                      |

## Limitations

- Susceptible to attrition: Worst threat to internal validity
	- If attrition is correlated with treatment, that's bad
	- If attrition is systematic, that's bad
- Noncompliance
	- Individuals assigned to treatment may not take it
	- Individuals assigned to control may take treatment
	- Intent-to-Treat vs Treatment-on-the-Treated
- Not always good at external validity
- Without understanding the various [Effect Modifiers](## Effect Modifiers), we will get wrong inferences
	- because you will **mistake** a local effect for a global effect that applies for all scenarios. Hence, there are limits for Random testing without understanding the causal mechanism. This clearly disproves the thinking that “RCTs are the golden standard for causal inference”
	- Only [Conditional Randomized Experiments](#Conditional Randomized Experiments) give correct readings, because it helps obtain the true causality without effect of any other factors. For eg, a lot of Psychology studies are performed on Psychology students, hence it doesn’t really give true research findings.
- nearly impossible to perform random tests in economics, due to the following
	- **infeasible** (govt/monetary policies)
	- **ethical** reasons (smoking - lung cancer)
	- **cost**
	- **duration**  (childhood intervention & adult outcomes)
	  - Long duration studies often suffer from significant (non-random) attrition
- Hinderances
  - RCTs require special conditions if they are to be conducted successfully
  - local agreements
  - compliant subjects
  - affordable administrators
  - multiple blinding
  - people competent to measure and record outcomes reliably
- High dimensional treatment/nuisance factors
  - possibility of too many known/unknown **effect modifiers**
  - if we do not control for the effect modifiers, the causal effect estimate obtained will be very local. This limits the usefulness of study
- [**Scaling-up**](#Scaling-Up of RT) of effects to the population may give opposite results of the RCT sample
  - Predicting the same results at scale as in the trial can be problematic, as the larger target population can be very different from the study population, so the causal effects may not be transportable
  - **General equilibrium effects**
    - even if the trial sample is a random sample of the target population, so that the target population $\sim$ the study population, applying the same intervention to everyone in the population could generate very different effects than in the trial
    - Hence, the result we obtain is a local result conditional to the current equilibrium
  - Violation of **SUTVA** (Stable Unit Treatment Value Assumption)
    - SUTVA = Assumption that individual’s potential outcome under a treatment does not depend on the treatments received by other individuals, as there is an assumption that there is **no interaction b/w individuals**.
    - SUTVA can be thought of as an i.i.d. assumption on causal effects
    - If the causal effect depends on how many individuals receive the treatment, then SUTVA is violated. Treatment dilution: treatment is less effective as more people get it
    - In the scaling up effects explanation, we can see that market equilibrium is affecting the outcomes


### Scaling-Up of RT

Govt policy to increase farmers’ incomes through subsidized fertilizers, based on effectiveness in RCT. Increased production for the sample farmers would increase their revenue, but if all farmers used this fertilizer, then the overall supply would increase. Assuming that the demand for the produce is inelastic, then the price would reduce. Hence, **the income of the farmers would actually reduce**. Therefore, the policy of encouraging all farmers to use fertilizers would be bad.

The same thing goes for **effect of education on earnings**. If everyone is now educated, the supply for high-skilled labor increases but the demand is still the same, hence its value decreases.

### SUTVA Violation

- Violation of SUTVA can also be viewed as a problem of ill-defined interventions
- When SUTVA is violated
  - Only slight violation can be tolerated
  - the sample treatment and the population treatment are essentially different interventions
  - If violated, then we need to take the interaction into account

#### Handling

Others receiving the treatment must be considered as an effect modifier of the Randomized Test.

- Learn $p( \ y_i \vert \text{do}(x_i) \ )$ or $p( \ y_i \vert \text{do}(x_i), x_j \ )$ treating $x_j$ as an effect modifier
  - When estimating the treatment effect on a individual, if SUTVA is violated then we need to consider the treatments received by other individuals as effect modifiers
- Learn $p( \ y_i \vert \text{do}(x_i, x_j) \ )$
  - This requires changing the unit of analysis from the original individual to a population of those units where interaction occurs and is confined in
    - We can define each individual $i$ to be a “local population” where such interference occurs and is confined in, and let the underlying population be a population of such local populations

#### Why does violation happen a lot?

Unlike medical sciences, socio-economic outcomes are often results of individual interaction. If the market is not perfectly-competitive, individual choices are rarely independent and each person’s choice affects other people.

| Scale |                                                              |
| ----- | ------------------------------------------------------------ |
| Micro | Social/Strategic Interaction<br />(Firm competition in oligopolistic markets) |
| Macro | General Equilibrium effects                                  |

However, it is negligible in certain cases: buyers and sellers in competitive markets

## Examples of RCT

### Exams

For eg, if there are 2 exam sets. The *treatments* are

- $x = 0 \to$ easy set
- $x = 1 \to$ hard set

| Scenario                                   | Variable | Comment                                            |                                                                                                           |
| ------------------------------------------ | -------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Asking students to volunteer for hard test | $E(y     | x = 1)$                                            | Only a certain type of students will volunteer to do so [Self-Selection Effect](## Self-Selection Effect) |
| Forcing everyone to take the hard test     | $E(y^1)$ | Everyone has received the input (taking hard test) |                                                                                                           |
| Randomly assigning sets                    | $E(y     | x=1) = E(y^1)$                                     | The population and treated sample will **very likely** have the same type of people                       |

### Demand Estimation

#### Goal

to know consumer demand for a product wrt price

#### Given Data

$$
D = \{
(p_1, q_1), \dots, (p_n, q_n)
\}
$$

- $q_i = Q_i^L \cdot (p_i==L) \ + \ Q_i^H \cdot (p_i==H)$
- $\{ (p_1, Q_1^L, Q_1^H), \dots, (p_n, Q_n^L, Q_n^H) \} \overset{\text{iid}}{\sim} P(p, Q^L, Q^H)$

#### Method

- treatment - price $p$
- outcome - purchases $q$

Let’s assume there are only 2 inputs (price level) - $p \in \{ L, H \}$. Then $\exists$

- 2 potential outcomes (demand level) - $q \in \{ Q^L, Q^H \}$
- Desired causal effect

$$
\text{ATE} = E[Q^L - Q^H]
$$

From the data, we can learn $P(q \vert p = a), a \in L, H$

#### Problem with observational data

We cannot directly use P and Q to estimate the demand/supply function.
This is because every data point is an equilibrium point and cannot be taken as the demand/supply curve. So, self-selection effect comes into play.

Without exchangeability, $P(Q^a) \ne P(Q^a \vert p=a) = P(q \vert p=a)$

The group that “received” the treatment $p=L$ could be systematically different than the group that “received” $p=H$

- People that buy when price is high can be richer than those who buy when price is low
- If we observe the person over time, then their income may be different when the price is low vs price is high

Hence, there will be effect of income elasticity which will alter our understanding the price elasticity

Changing price will **not** help in determination of the causal effect of price change.
$$
\text{ATE} \ne E(q|p = l) - E(q|p = h)
$$
This is because, in real life, $p$ is **not** randomly-assigned. So, the people who buy at high price and low price are completely-different; the populations are different in both the cases. So, the effect of income comes into picture. Therefore, the true and direct causal effect of price will not be understood.

#### Solution

Companies could run experiments by randomly assigning prices to customers in different markets and over time

Companies could perform A/B testing by running experiments by randomly assigning prices in different markets and over time. This change in price will target the individual, so the true treatment effect will be learnt, as the income of people is quite constant.

This is mainly used by online companies, as it is inexpensive to do so.

### Giffen Behavior

cannot be trusted through observational data. This is because

Higher prices are often associated with more purchases, but is it

- higher demand causing both higher prices and more purchases, or
- higher prices causing people to buy more (Giffen behavior)

The prices are “chosen”, so the analysis using observed increased prices is not necessarily a treatment and does not help us obtain the true causal effect

We need to keep in mind

1. inflation
2. increase in wages

However, with randomized treatment (such as subsidies for the commodity), we can derive the true giffen behavior and hence make correct analysis.

### Classroom Size

#### Goal

to know the effect of classroom size on student performance

#### Given Data

$$
D = \{
(p_1, q_1), \dots, (p_n, q_n)
\}
$$

#### Method

- treatment - classroom size
- outcome - student performance

Let’s assume there are only

- 2 inputs (room size) - $s \in \{ S, L \}$
- 2 potential outcomes (performance) - $p \in \{ p^S, p^L \}$

$$
\text{ATE} = E[p^L - p^S]
$$

#### Problem

Weaker students often deliberately grouped into smaller classes

Hence, Many studies of education production using non-experimental data suggest there is little or no link between class size and student learning

#### Solution

Randomized assignment of classroom size is necessary, as usually the more well-off students will be in private schools , so clearly there is self-selection effect here. So we will take the same group of kids, and randomly assign a small and large room.

#### Study in Tennesee, USA

Random assignment of classroom sizes to students showed that students in smaller classrooms performed better.

However, this may conclusion may **not** be accurate for the state of Tennesee itself, because there could be some other factors in play here. **Maybe** the students are not accustomed to the new large classrooms, which affects the performance. So over time, difference in performance might be nothing the more the students get used to it.

This result definitely can**not** be used directly elsewhere. This is because the composition of tested sample and the other populations will be different:

- income
- race
- culture

#### Solution

[Causal Mechanism Learning](06 Mechanism.md)

In this case, let’s analyze the following: 
> How would a small class help in performance?

hmm… because, students sit closer to the board. This mechanism is applicable everywhere. If this is the only mechanism, then we can confidently say that the results of the Tennessee experiment can be applied everywhere.

For example, maybe smaller classrooms enable easier interactions between students and teachers. Hence, smaller classrooms helpful in the US/Europe where they have a lot of group work and interactions; but not in India/China as the teacher mostly just teaches without much interactions.

### Fertilizer

#### Goal

to know the effect of fertilizer on crop output

#### Method

- treatment - using fertilizer
- outcome - crop output

#### Problem

The result might not be accurate with just randomized experiment. This is because, the effectiveness of fertilizer depends on the temperature (effect modifier)

#### Solution

[Conditional Randomized Experiment](## Conditional Randomized Experiment)

Randomized experiments at different temperatures

- a randomized experiment at low temperature
- a randomized experiment at high temperature

This is because, the temperature affects the causal effect of fertilizer on the crop output. By independent testing at different temperatures, we can keep the effect of temperature as a constant, and we will get the true causal effect of fertilizer.

### Fumigation and Yield

Fumigation is the use of fumigants to control eelworms which affects crop yield

Suppose, in a place A, we conduct an RCT to study the effect of fumigation on yield by randomly selecting $N$ barley fields and randomly applying fumigation to $M$ of them. The result shows that fumigation increases barley yield by 20%.

The understanding of the result depends on our understanding of the causal mechanism and its implied effect modifiers.

We need to investigate the effect of

- season when the study was performed
- previous year’s crop on that same field
- other prior known/hypothesised effect modifiers

For eg:
$$
\begin{aligned}
&E(\text{Causal effect} \vert \text{Summer}, \text{Same Crop}) \\
\ne & E(\text{Causal effect} \vert \text{Winter}, \text{Same Crop}) \\
\ne & E(\text{Causal effect} \vert \text{Summer}, \text{Alternated Crop}) \\
\ne & E(\text{Causal effect} \vert \text{Winter}, \text{Alternated Crop})
\end{aligned}
$$
Hence, this causal effect is **local**

### Psychology Studies

Studies of most Psychology studies are WEIRD: Western, Educated, Industrialized, Rich, Democratic, particularly American undergrads

## When best to do RCT

- Demand for treatment exceeds supply
- Treatment will be phased in over time
- Treatment is in equipoise (genuine uncertainty)
- Local culture open to randomization
- Monopolist system: A/B testing for online services

## When not to do RCT

- When it is unethical or illegal
- When you need immediate results
- When you want to measure something that happened in the past
- When it involves universal ongoing phenomena (like pandemics)

## A/B Testing

- hashing of `user_id`

### A/B Testing vs A/A Testing

A/A test 
- ensure if testing system is working correctly
- quantify the amount of variation that can occur between two identical groups naturally