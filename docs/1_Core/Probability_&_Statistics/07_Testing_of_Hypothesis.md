# Testing of Hypothesis

$\alpha$

- level of significance
- size of critical region

Confidence level = $(1-\alpha) \times 100 \%$

The entire distribution is divided into 2 regions

1. Critical Region
   Region of **rejection** of $H_0$
   it is decided based on $H_1$
2. Acceptance Region
   Region of **acceptance** of $H_0$

## Population Mean

$$
\begin{aligned}
H_0: \mu &= \mu_0 & &\text{(Null Hypothesis)} \\
H_1: \mu &< \mu_0, \mu \ne \mu_0, \mu > \mu_0 & &\text{(Alternative Hypothesis)} \\
\end{aligned}
$$

| $\sigma^2$ |   $n$    |    Test Statistic/Probability Distribution     |
| :--------: | :------: | :--------------------------------------------: |
|   known    |   any    | $z_c = \frac{\bar x - \mu_0}{\sigma/\sqrt n}$ |
|  unknown   |  $>30$   |  $z_c = \frac{\bar x - \mu_0}{s/ \sqrt n}$   |
|  unknown   | $\le 30$ |  $t_c = \frac{\bar x - \mu_0}{s / \sqrt n}$  |

### Critical Region

|         |                         Left-Tailed                          |                          Two-Tailed                          |                         Right-Tailed                         |
| :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  $H_1$  |                        $\mu < \mu_0$                         |                       $\mu \ne \mu_0$                        |                        $\mu > \mu_0$                         |
| p-value |              $F(z_c)$ <br /> $\alpha(t-\text{dist})$               |           $2[ F(-z_c) ]$ <br /> $2 \alpha(t-\text{dist})$           |              $F(-z_c)$ <br /> $\alpha(t-\text{dist})$              |
|  Cases  | Accept $H_1$ if <br />$\begin{aligned} z_c & \le -z_\alpha \\ t_c &\le -t_{(n-1), \alpha} \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned} z_c \le -z_{\alpha/2} &\text{ or } z_c \ge +z_{\alpha/2}\\ t_c \le -t_{(n-1), (\alpha/2)} &\text{ or } t_c \ge +t_{(n-1), (\alpha/2)} \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned} z_c &\ge +z_\alpha \\ t_c &\ge +t_{(n-1), \alpha} \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ |

## Proportion

$$
\begin{aligned}
H_0: p &= p_0 & &\text{(Null Hypothesis)} \\
H_1: p &< p_0, p \ne p_0, p > p_0 & &\text{(Alternative Hypothesis)} \\
z_c &= \frac{\hat p - p_0}{
	\sqrt{ \frac{p_0(1-p_0)}{n} }
} & & \hat p = \frac x n = \text{Estimated value of } p\\
\end{aligned}
$$

### Critical Region

|         |                         Left-Tailed                          |                          Two-Tailed                          |                         Right-Tailed                         |
| :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  $H_1$  |                          $p < p_0$                           |                         $p \ne p_0$                          |                          $p > p_0$                           |
| p-value |                           $F(z_c)$                           |                        $2[ F(-z_c) ]$                        |                          $F(-z_c)$                           |
|  Cases  | Accept $H_1$ if <br />$\begin{aligned}z_c &\le -z_\alpha \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned} z_c \le -z_{\alpha/2} &\text{ or } z_c \ge +z_{\alpha/2} \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned} z_c &\ge +z_\alpha \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ |

## Variance/SD

$$
\begin{aligned}
H_0: \sigma^2 &= \sigma^2_0 & &\text{(Null Hypothesis)} \\
H_1: \sigma^2 &< \sigma^2_0, \sigma^2 \ne \sigma^2_0, \sigma^2 > \sigma^2_0 & &\text{(Alternative Hypothesis)} \\
\chi_c^2 &= (n-1) \frac{s^2}{\sigma_0^2}
\end{aligned}
$$

### Critical Region

|         |                         Left-Tailed                          |                          Two-Tailed                          |                         Right-Tailed                         |
| :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  $H_1$  |                          $p < p_0$                           |                         $p \ne p_0$                          |                          $p > p_0$                           |
| p-value |                     1 - $\alpha$(table)                      |                     1 - $\alpha$(table)                      |                     1 - $\alpha$(table)                      |
|  Cases  | Accept $H_1$ if <br />$\begin{aligned}\chi_c^2 &\le \chi^2_{(n-1), (1-\alpha)}  \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned}\chi_c^2 \le \chi^2_{(n-1), (1-\alpha/2)} &\text{ or } \chi_c^2 \ge \chi^2_{(n-1), (\alpha/2)} \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ | Accept $H_1$ if <br />$\begin{aligned}\chi_c^2 &\ge \chi^2_{(n-1), \alpha}  \\ p &\le \alpha \end{aligned}$<br /><br />else accept $H_0$ |

## Errors

|              | $H_0$ is true           | $H_0$ is false         | $H_0$ is incorrect                                   |
| ------------ | ----------------------- | ---------------------- | ---------------------------------------------------- |
| Reject $H_0$ | Type 1 Error = $\alpha$ | Correct                | Type 3 Error<br />Right answer to the wrong question |
| Accept $H_0$ | Correct                 | Type 2 Error = $\beta$ |                                                      |

Type 1 error is alright, but Type 2 error is dangerous

- $\alpha$ = P(reject $H_0$ | $H_0$ is true)
- $\beta$ = P(accept $H_0$ | $H_0$ is false)

## Power of Test

$$
\text{Power of Test} = 1 - \beta
$$

Greater the power of test, the better
means that we can more accurately detect when $H_0$ is false

