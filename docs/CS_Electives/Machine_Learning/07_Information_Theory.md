# Information Theory

Information entropy is developed to describe the avg amount of info needed to specify the state of a RV

Quantifies how much **new** information about a RV $x$ is obtained when we observe a specific value $x_i$

- Depends on ‘degree of surprise’: highly improbable value conveys more information than likely one
- If we know an event is certain to happen, we would receive no information when we actually observe it happening

Let $h(x)$ denote the information content of an event $x$, such that

- $p(x) \propto \dfrac{1}{p(x)}$
- For 2 independent events $A$ and $B$

$$
\begin{aligned}
p(A \land B) &= p(A) \cdot p(B) \\
\implies h(AB) &= h(A) + h(B)
\end{aligned}
$$

$$
h(x) = \log \left \vert \dfrac{1}{p(x)} \right \vert
$$

## Shannon/Information Entropy

Entropy of a probability distribution $p(x)$ is the average amount of information transmitted by discrete random variable $x$
$$
\begin{aligned}
H(p)
&= E_p[h(x)] \\
&= \sum_x p(x) \log \left \vert \dfrac{1}{p(x)} \right \vert
\end{aligned}
$$

- Distributions with sharp peaks around a few values will have a relatively low entropy
- Those spread evenly across many values will have higher entropy

If we use $\ln$ instead of $\log$ for $H(p)$, then $H(p)$ is a lower bound on the avg no of bits needed to encode a RV with pdf $p$. Achieving this bound requires using an optimal coding scheme designed for $p$, which assigns

- shorter codes to higher probability events
- longer codes to less probable events

## Cross Entropy

The average amount of information needed to specify $x$ as a result of using pdf $q(x)$ instead of true $p(x)$

If we use $\ln$ instead of $\log$, it is the average no of bits needed to encode a RV using a coding scheme designed for $q(x)$ instead of true $p(x)$
$$
\begin{aligned}
H(p, q)
&= E_p \left[ \log \left \vert \dfrac{1}{q(x)} \right \vert \right] \\
&= \sum_x p(x) \log \left \vert \dfrac{1}{q(x)} \right \vert
\end{aligned}
$$

## KL Divergence

Dissimilarity measure of 2 probability distributions

Represents the avg **additional info** required to
 specify $x$ due to using $q(x)$ instead of true $p(x)$
$$
\begin{aligned} 
D_\text{KL}(p \vert \vert q)
&= \text{Cross Entropy} - \text{Entropy} \\
&= H(p, q) - H(p) \\
& = \sum_x \log \left \vert \dfrac{p(x)}{q(x)} \right \vert \cdot p(x) & \text{(Discrete)} \\
& = \int_x \log \left \vert \dfrac{p(x)}{q(x)} \right \vert \cdot p(x) & \text{(Continuous)} \\
D_\text{KL}(p \vert \vert q) &\ge 0
\end{aligned}
$$
Note: KL divergence is not symmetric $\implies D_\text{KL}(p \vert \vert q) \ne D_\text{KL}(q \vert \vert p)$. Hence it is not a proper distance measure

Given a **fixed** distribution $p$, optimizing for a $q$ for the following 3 goals are equivalent

- minimize $D_\text{KL}(p \vert \vert q)$
- minimize $H(p, q)$
- maximize $L(q \vert x)$, where $x \sim p(x)$
