# Causal Mechanism Learning

To understand the true meaning and scope of a causal effect, we need to understand the underlying causal mechanism, based on **prior knowledge** - information and analyses.

This is important to understand

- causal effect - what it means, where it applies
- transportability of results

We have to understand the true mechanism and reason why a treatment affects the outcome. A randomized experiment is not a suitable reason to skip that step.

If the mechanisms in play are globally-applicable, then we can conclude that results of the study can be applied everywhere. Otherwise, we can**not** confidently conclude that.

Blindly following randomized experiments approach, without understanding the underlying mechanism, is incorrect.

# Examples

## Russel’s Chicken

This short story shows how pure reliance on past data is bad.

The chicken assumes that whenever the farmer comes, it is to feed it. However, there will one day, the farmer comes to kill it.

Hence, the lack of understanding **why** something happens might be very dangerous.

## Simpson’s Paradox

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

## US Political Support

Similar to [Simpson’s Paradox](#Simpson’s Paradox)

[Aggregate Reversal](#Aggregate Reversal)

| Level      | Richer you are, more likely to be a __ | Reason                                                       |
| ---------- | -------------------------------------- | ------------------------------------------------------------ |
| Individual | Republican                             | Republicans are richer and want lower taxes                  |
| State      | Democrat                               | Richer societies are usually morally ‘modern’; Poorer one are usually conservative and religious<br /><br />Democrats have more ‘modern’ policies |

## Survivorship Bias

This wasn’t taught in this course, but i just remembered.

The planes that returned from war had lot of spots with bullet shots.

Some person suggested strengthening only those spots. Initially, that makes sense - these are the areas that got shot so we need to strengthen. But, that is wrong.

Another person said that these are the planes that returned **despite** getting shot at these spots. That means that we have to focus on other places, because the planes that got shot there never returned.

Clearly data can be misleading, without understanding the underlying cause.

# Aggregate Reversal

If you just look at statistical data, it might be misleading.

Once we devide the population into sub-population based on categories such as sex, then it becomes clearer. This is because why try understanding the underlying mechanism. This phenomenon is called as **aggregate reversal**.

Hence, statistical correlation is unstable. This is why causal inference is important.