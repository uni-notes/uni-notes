## Sampling

is used when it is not feasible to analyse the entire population

## Population v Sample

| Property           |                          Population                          |                            Sample                            |
| ------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Definition         | comprises of all units pertaining to a particular characteristic under study | is a part of a population, which is selected such that it is representative of the entire population |
| Size               |                             $N$                              |                             $n$                              |
| Mean               |                            $\mu$                             |               $\bar x = \dfrac {\sum x_i}{n}$                |
| Variance           |                          $\sigma^2$                          | $s^2 = \dfrac 1 {n-1} \left[ \sum {x_i}^2 - \dfrac { \left(\sum x_i \right)^2}{n} \right]$ |
| Standard Deviation |                           $\sigma$                           |                             $s$                              |

### Relations

$$
\begin{align}
E(\bar x) &= \mu, E(s^2) = \sigma^2 , E(s) = \sigma \\
s^2 &= \frac{\sigma^2} {n} , s = \frac{\sigma} {\sqrt n} \\
z_\text{sample} &= \frac {\bar x - \mu}{ \sigma/\sqrt n }
\end{align}
$$

## Central Limit Theorem

PDF for any sample with $n>30$ tends to normal distribution

$$
\bar x \sim N \left(\mu, \frac{\sigma^2}{n} \right)
$$

## Estimation

Using the sample, we estimate population parameter(s)

## Interval Estimation

Confidence % $= 1- \alpha$

Most common is $95\%$ confidence interval estimate

$$
\begin{align}
1 - \alpha &= 0.95 \\\alpha &= 0.05 \\\alpha/\small 2 &= 0.025
\end{align}
$$

### Population mean

| $\sigma^2$ | $n$   | statistic | $\mu$ |
| :-------------------------------: | :---: | :-------------------------------: | :------: |
| known | any   | $z = \dfrac {\bar x - \mu} {\sigma / \sqrt n}$ | $\bar x \pm z_{\alpha/\small 2} \cdot \dfrac \sigma {\sqrt n}$ |
| unknown | $>30$ | $z = \dfrac {\bar x - \mu} {s/ \sqrt n}$ | $\bar x \pm z_{\alpha/\small 2} \cdot \dfrac s {\sqrt n}$ |
| unknown | $\le 30$ | $t = \dfrac {\bar x - \mu} {s / \sqrt n}$ | $\bar x \pm t_{\small n-1, \alpha/\small 2} \cdot \dfrac s {\sqrt n} \\(n-1) \to \text{deg of freedom}$ |

$$
\begin{align}
n &= \left( \frac{z_{\alpha/\small 2} \cdot \sigma}{w} \right)^2 \\&= \left( \frac{z_{\alpha/\small 2} \cdot s}{w} \right)^2
\end{align}
$$

where

- $n$ is sample size
- $w$ is distance from $\mu$ = $\frac{\text{interval width}}{2}$

### Proportion

$$
\begin{align}
p &= \hat p \pm z_{\alpha/\small2} \sqrt {\frac{\hat p \hat q}{n}} \\\hat p &= \frac x n = \frac{\text{Favorable no of cases}}{\text{Total no of cases}} \\\hat q &= 1 - \hat p
\end{align}
$$

### Population Variance / SD

$$
\begin{align}
\sigma^2 &= \left[
\frac {(n-1)s^2}{\chi^2_{(n-1), (\alpha/\small 2)}},
\frac {(n-1)s^2}{\chi^2_{(n-1), (1-\alpha/\small 2)}}
\right] \\
\sigma &= \sqrt {\sigma^2}
\end{align}
$$

