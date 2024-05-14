# Gaussian EM K-Means Clustering

Probabilistic clustering which requires K means as well; the output of k means is fed into this model

## EM for Gaussian Mixture

Alternates between 2 steps

1. E-Step: Given an estimate $\theta_e$ of the weights, compute $P_\theta(y \vert x_i)$ and use it to ‘hallucinate’ expected cluster assignments $z_i$
2. M-Step: Find a new $\theta_{e+1}$ that maximizes the marginal log-likelihood by optimizing $P_\theta(x_i, z_i)$ given the $z_i$ from step 1

$$
\begin{aligned}
\theta_{e+1}
&= \arg \max_\theta \sum_{i=1}^n \mathbb E_{z_i \sim P_{\theta_e} (y \vert x_i)} \log P_\theta(x_i, z_i) \\
&= \arg \max_\theta \sum_{i=1}^n \sum_{c=1}^C P_{\theta_e}(y=c \vert x_i) \log P_\theta(x_i, y=k)
\end{aligned}
$$

$$
\begin{aligned}
P_\theta(y=c \vert x)
&= \dfrac{P_\theta (y=c, x)}{P_\theta(x)} \\
&= \dfrac{P_\theta (y=c) P_\theta (x \vert y=c)}{
\sum_{c=1}^C P_\theta (y=c) P_\theta (x \vert y=k)
}
\end{aligned}
$$

3. Go to step 1

This process increases the marginal likelihood at each step and eventually converges

### Advantages

- Easy to implement
- Guaranteed to converge?
- Works for many ML models

### Limitations

- Can get stuck in local optima
- May not be able to compute $P_{\theta_e} (y \vert x_i)$ in each model

## Steps

Expectation Maximization

1. Initialize gaussian parameters $\mu_k, \Sigma_k, \pi_k$

   1. $\mu_k \leftarrow \mu_k$
   2. $\Sigma_k \leftarrow \text{cov $\Big($ cluster($k$) $\Big)$}$
   3. $\pi_k = \frac{\text{No of points in } k}{\text{Total no of points}}$

2. E Step: Assign each point $X_n$ an assignment score $\gamma(z_{nk})$ for each cluster $k$

$$
\gamma(z_{nk}) = \frac{
\pi_k N(x_n|\mu_k, \Sigma_k)
}{
\sum_{i=1}^K \pi_i N(x_n|\mu_i, \Sigma_i)
}
$$

3. M Step: Given scores, adjust $\mu_k, \pi_k, \Sigma_k$ for each cluster $k$

$$
\begin{aligned}
\text{Let } N_k &= \sum_{n=1}^N \gamma(z_{nk}) \\
N &= \text{Sample Size} \\
\mu_k^\text{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n \\
\Sigma_k^\text{new} &= \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^\text{new}) (x_n - \mu_k^\text{new})^T \\
\pi_k^\text{new} &= \frac{N_k}{N}
\end{aligned}
$$

4. Evaluate log likelihood

$$
\ln p(X| \mu, \Sigma, \pi) =
\sum_{n=1}^N
\ln \left|
\sum_{k=1}^K \pi_k N(x_n | \mu_k, \Sigma_k)
\right|
$$

5. Stop if likelihood/parameters converge

## K Means vs Gaussian Mixture

|                        | K means                                                      | Gaussian Mixture                                             |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                        | Classifier model                                             | Probability model                                            |
| Probabilistic?         | ❌                                                            | ✅                                                            |
| Classifier Type        | Hard                                                         | Soft                                                         |
| Pamater to fit to data | $\mu_k$                                                      | $\mu_k, \Sigma_k, \pi_k$                                     |
| ❌ Disadvantage         | If a class may belong to multiple clusters, we have to assign the first one<br />If sample size is too small, initial grouping determines clusters significantly | Complex                                                      |
| ✅ Advantage            | Simple<br />Fast and efficient, with $O(tkn),$ where<br />- $t =$ no of iterations<br/>- $k =$ no of clusters<br/>- $n =$ no of sample points | If a class may belong to multiple clusters, we have a probability to back it up |
