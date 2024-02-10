# Gaussian Classifier

Used for classifying continuous data

$$
\begin{aligned}
P(C|x)
& \propto P(x|C) \times P(C) \\
& \propto N(x; \mu_c, \sigma^2_c) P(C) \\
\underbrace{P(C)}_{\text{Posterior}}
& \propto
\underbrace{
\frac{1}{\sqrt{2\pi \sigma^2_c}}
\ \exp \left(
\frac{-(x-\mu_c)^2}{2\sigma^2_c}
\right)
}_{\text{Likelihood}}
\underbrace{P(C)|x}_{\text{Prior}}
\end{aligned}
$$

However equation is not used as it is; we take $\log$ on both sides and find **log likehood**

$$
\begin{aligned}
\text{LL}(x|C)
&= \text{LL}(x|\mu_c, \sigma_c^2) \\
&= \ln P(x | \mu_c, \sigma_c^2) \\
&= \ln \left[
\frac{1}{\sqrt{2 \pi \sigma_c^2}}
\right]
 \ \exp somethign
\end{aligned}
$$

$$
\text{LL} \underbrace{(C|x)}_\text{Posterior} =
\text{LL} \underbrace{(x|C)}_\text{Likelihood} +
\text{LR} \underbrace{(C)}_\text{Posterior}
$$

### 2 Classes

$$
\begin{aligned}
\ln \frac{P(C_1 | x)}{P(C_2 | x)}
&= \ln P(C_1 | x) - \ln P(C_2 | x) \\
&= \frac{-1}{2} ()
\end{aligned}
$$

- If log ratio $\ge 0$, assign to $C_1$
- If log ratio $<0$, assign to $C_2$

We need to ensure that we have equal sample of both classes, so that the prior probabilities of both the classes in the formula is the same.

