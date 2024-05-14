# Autocorrelation



## Durbin-Watson Test

### Positive

- $H_0: \rho = 0$
- $H_1: \rho > 0$ (not negative)

$$
\begin{aligned}
D
&= \dfrac{
\sum_{i=p}^n (u_i - u_{i-p})^2
}{
\sum_{i=1}^n (u_i)^2
} \\
& \approx 2(1-\rho)
\end{aligned}
$$

where $p=$ lag being tested

- If $D>D_h$, cannot reject null hypothesis
- If $D<D_l$, reject null hypothesis
- If $D_l < D<D_h$, inconclusive



- $D \ge 2 \implies$ no autocorrelation
- $D \to 0 \implies$ perfect autocorrelation

### Negative

$D' = 4-D$

## Runs Test

Run: any sequence on the same side of 0

Usually one-tailed to test for +ve correlation

- +ve correlation: bounces less frequently
- -ve correlation: bounces very frequently; (not very common in data)

$$
\begin{aligned}
\bar R
&= \dfrac{2 n_+ n_-}{n} + 1 \\
s^2_R
&= \dfrac{2 n_+ n_- (2 n_+ n_- - n)}{n^2 (n-1)} \\
\implies Z_R &= \dfrac{R-\bar R}{s_R} \sim N(0, 1)
\end{aligned}
$$

where

- $R=$ number of runs in data
- $n_+=$ number of +ve residuals
- $n_-=$ number of -ve residuals
- $n=$ total number of residuals