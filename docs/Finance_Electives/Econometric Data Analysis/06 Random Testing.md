# Randomization/Randomized Tests

ensures that correlation = causation

helps estimate counterfactual outcome by ensuring

- independence/exchangeability of $x$
- similarity of population and treated sample

$$
\text{ATE = ATT = ATU} \notag
$$

## Independence

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

## Similarity

To ensure that $E(y|x=1) = E(y^1)$, we need to ensure that the population and treated-sample are the similar in their features.

## Without Randomization

There will be statistical correlation without causal correlation (to be avoided). This is due to [Self-Selection Effect](#Self-Selection Effect). So the selection will be biased.

$$
\begin{aligned}
& E(y | x = 1) - E(y | x = 0) \\
=& 
\underbrace{
	E( y|x = 1) \textcolor{orange} {- E(y^0|x=1)}
}_\text{ATT}
+
\underbrace{
\textcolor{orange}{ E(y^0|x=1) } - E(y | x = 0)
}_{ \ne \text{ATU} }
\\

\ne & E(y' - y^0) (\text{Unrandomized})
\end{aligned}
$$

## Derivations from Randomized Tests

| Property                         | Meaning                                            |
| -------------------------------- | -------------------------------------------------- |
| Association (Correlation)        | $P(y|x=a) \ne P(y|x=b)$                            |
| Causation                        | $P(y^a) \ne P(y^b)$                                |
| Correlation $\implies$ Causation | Random assignment of $x \implies (y^a) = P(y|x=a)$ |

## Limitations

Without understanding the various [Effect Modifiers](# Effect Modifiers), we will get wrong inferences, because you will **mistake** a local effect for a global effect that applies for all scenarios. Hence, there are limits for Random testing without understanding the causal effects.

Only [Conditional Randomized Experiments](#Conditional Randomized Experiments) give correct readings, because it helps obtain the true causality without effect of any other factors. For eg, a lot of Psychology studies are performed on Psychology students, hence it doesn’t really give true research findings.

Also, it’s nearly impossible to perform random tests in economics, due

- **infeasible** (govt policies)
- **ethical** reasons (smoking - lung cancer)
- **cost & duration** (childhood intervention & adult outcomes)
- possibility of too many known/unknown **effect modifiers**
- [**Scaling-up**](#Scaling-Up of RT) of effects to the population may give adverse results
- [Violation of **SUTVA**](#Violation of SUTVA)

## Scaling-Up of RT

Let’s say the govt’s objective was to increase farmers’ incomes, and the RCT showed that the fertilizer was effective. Should the govt encourage all farmers to use this fertilizer?
In this case, increased production for the sample farmers would increase their revenue, but if all farmers used this fertilizer, then the overall supply would increase. Assuming that the demand for the produce is inelastic, then the price would reduce. Hence, **the income of the farmers would actually reduce**. Therefore, the policy of encouraging all farmers to use fertilizers would be bad.

The same thing goes for **effect of education on earnings**. If everyone is now educated, the supply for high-skilled labor increases but the demand is still the same, hence its value decreases.

## Violation of SUTVA

Stable Unit Treatment Value Assumption

Only slight violation can be tolerated. (Find out how much can be tolerated)

In the scaling up effects explanation, we can see that market equilibrium is affecting the outcomes. This violates the assumption of randomized test that the potential outcome due to treatment given to each person is independent from others’ outcome, as there is interaction between individuals.

If the causal effect depends on how many indiduals received the treatment, then SUTVA is violated. This is called as treatment dilution - the treatment is less effective as more people get it.

### Handling Treatment Dilution

Others receiving the treatment must be considered as an effect modifier of the Randomized Test.

### Why does violation happen a lot?

Unlike medical sciences, socio-economic outcomes are often results of individual interaction. If the market is not perfectly-competitive, individual choices are rarely independent and each person’s choice affects other people.

| Scale |                                                              |
| ----- | ------------------------------------------------------------ |
| Micro | Social/Strategic Interaction<br />(Firm competition in oligopolistic markets) |
| Macro | General Equilibrium effects                                  |



# Conditional Randomized Experiments

If one or more external parameters affect the causal effect of the treatment on the outcomes, then we have to do different randomized conditions.

By independent testing at different conditions, we can keep the effect of the external parameter as a constant, and we will get the true causal effect of the treatment.

It leads to **conditional exchangeability**
$$
x \perp \!\!\! \perp
(y^1, \dots, y^A) | s
$$
where $s$ is the sub-population. This is the [Effect Modifiers](# Effect Modifiers)

## Effect Modifiers

Effect modifiers/Nuisance factors are anything that change the causal effect of a treatment. Controlling them help control conditionally randomized experiments.

An example could be gender, temperature, etc.

Mathematically, $s$ is an effect modifier if
$$
P(y^1 - y^0) \ne P(y^1 - y^0 | s)
$$

So, if we now take the effect modifier into play,
$$
\text{ATE} = \int E[y^1 - y^0 \ | \ s] \cdot P(s) \cdot ds
$$

where $P(s)$ is the distribution of effect modifier.

Then, the result of the randomized test actually gives us $E[\tilde \tau]$, which may **not** be equal to $E[\tau]$

### Example

For example, the yield depends on the season and the crop for which was grown previous year. Let’s take the example of a a randomized test of a fertilizer used  in the summer.

If no crop was grown the previous year, then we 

| Crop Grown in Field Last Year | Result of Randomized Test $E[\tilde \tau]$                   |
| ----------------------------- | ------------------------------------------------------------ |
| No crop                       | $E[\tau | \text{Summer, No crop}]$                           |
| Rice only                     | $E[\tau | \text{Summer, Rice}]$                              |
| 50% Barley, 50% Rice          | $0.5 \cdot E[\tau |\text{Summer, Barley}] + 0.5 \cdot E[\tau |\text{Summer, Rice}]$ |

# Self-Selection Effect

In economics, we assume that everyone

- is rational
- makes decisions/selections to maximize self-interests

# Examples

## Exams

For eg, if there are 2 exam sets. The *treatments* are

- $x = 0 \to$ easy set
- $x = 1 \to$ hard set

| Scenario                                   | Variable            | Comment                                                      |
| ------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| Asking students to volunteer for hard test | $E(y | x = 1)$      | Only a certain type of students will volunteer to do so<br />[Self-Selection Effect](# Self-Selection Effect) |
| Forcing everyone to take the hard test     | $E(y^1)$            | Everyone has received the input (taking hard test)           |
| Randomly assigning sets                    | $E(y|x=1) = E(y^1)$ | The population and treated sample will **very likely** have the same type of people |

## Demand Estimation

### Goal

to know consumer demand for a product

### Problem with observational data

We cannot directly use P and Q to estimate the demand/supply function.
This is because every point is an equilbrium point. So, self-selection effect comes into play.

### Given Data

$$
D = \{
(p_1, q_1), \dots, (p_n, q_n)
\}
$$

### Method

- treatment - price
- outcome - purchase

Let’s assume there are only

- 2 inputs (price level) - $p \in \{ L, H \}$
- 2 potential outcomes (demand level) - $q \in \{ Q^L, Q^H \}$

$$
\text{ATE} = E[Q^L - Q^H]
$$

### Problem

But changing price will **not** help in determination of the causal effect of price change.
$$
\text{ATE} \ne E(q|p = l) - E(q|p = h)
$$
This is because, in real life, $p$ is **not** randomly-assigned. So, the people who buy at high price and low price are completely-different; the populations are different in both the cases. So, the effect of income comes into picture. Therefore, the true and direct causal effect of price will not be understood.

### Solution

Companies could perform A/B testing by running experiments by randomly assigning prices in different markets and over time. This change in price will target the individual, so the true treatment effect will be learnt, as the income of people is quite constant.

This is mainly used by online companies, as it is inexpensive to do so.

## Giffen Behavior

cannot be trusted through observational data. This is because

Higher prices are often associated with more purchases, but is it

- higher demand causing both higher prices and more purchases, or
- higher prices causing people to buy more (Giffen behavior)

We need to keep in mind

1. inflation
2. increase in wages

However, with randomization, we can analyze the giffen behavior and hence make correct analysis.

## Classroom Size

### Goal

to know the effect of classroom size on student performance

### Given Data

$$
D = \{
(p_1, q_1), \dots, (p_n, q_n)
\}
$$

### Method

- treatment - classroom size
- outcome - student performance

Let’s assume there are only

- 2 inputs (room size) - $s \in \{ S, L \}$
- 2 potential outcomes (performance) - $p \in \{ p^S, p^L \}$

$$
\text{ATE} = E[p^L - p^S]
$$

Randomized assignment of classroom size is necessary, as usually the more well-off students will be in private schools , so clearly there is self-selection effect here. So we will take the same group of kids, and randomly assign a small and large room.

### Study in Tennesee, USA

Random assignment of classroom sizes to students showed that students in smaller classrooms performed better.

However, this may conclusion may **not** be accurate for the state of Tennesee itself, because there could be some other factors in play here. **Maybe** the students are not accustomed to the new large classrooms, which affects the performance. So over time, difference in performance might be nothing the more the students get used to it.

This result definitely can**not** be used directly elsewhere. This is because the composition of tested sample and the other populations will be different:

- income
- race
- culture

### Solution

[Causal Mechanism Learning](06 Mechanism.md)

In this case, let’s analyze the following: 
> How would a small class help in performance?

hmm… because, students sit closer to the board. This mechanism is applicable everywhere. If this is the only mechanism, then we can confidently say that the results of the Tennessee experiment can be applied everywhere.

For example, maybe smaller classrooms enable easier interactions between students and teachers. Hence, smaller classrooms helpful in the US/Europe where they have a lot of group work and interactions; but not in India/China as the teacher mostly just teaches without much interactions.

## Fertilizer

### Goal

to know the effect of fertilizer on crop output

### Method

- treatment - using fertilizer
- outcome - crop output

### Problem

The result might not be accurate with just randomized experiment. This is because, the effectiveness of fertilizer depends on the temperature.

### Solution

[Conditional Randomized Experiment](# Conditional Randomized Experiment)

We have to do randomized experiments at different temperatures. This is because, the temperature affects the causal effect of fertilizer on the crop output. By independent testing at different temperatures, we can keep the effect of temperature as a constant, and we will get the true causal effect of fertilizer.

So we have to do

- a randomized experiment at low temperature
- a randomized experiment at high temperature