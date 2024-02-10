# Continuous

## Challenge

How to describe probability distribution

## Brownian Motion/Wiener Process

Basically a continuous version of simple RW: ‘Limit’ of simple RW

Denoted using $y_t = B(t)$

### Properties

- Always starts at 0: $P( y_0 = 0 ) = 1$
- Stationary $\forall s \in [0, t)$
  - $y_t - y_s \sim N(0, t-s)$
    - where $(t-s)$ is the length of the interval
- Independent increment: If intervals $[s_i, t_i]$ are non-overlapping, then $y_{t_i} - y_{s_i}$ are independent

### Characteristics

- Cross the independent axis indefinitely-often
- Does not deviate too much from $y_t = \sqrt{t}$
- Not differentiable
  - Standard calculus cannot be applied
  - Requires Ito calculus
- Max series 

$$
\begin{aligned}
& M_t = \max_{s \le t} (y_s) \\
\implies &P(M_t > a) = 2 \cdot P (y_t > a) \\
& \forall \ t, a > 0
\end{aligned}
$$

- Quadratic variation

$$
t = \frac{i}{n} T \\
\implies
\lim_{n \to \infty} \sum_{i=1}^n
(y_{t} - y_{t-1})^2
= T \\
\forall T > 0 \\

\implies (dB)^2 = dt
$$

## Implications

$$
\dfrac{d y_t}{y_t} = d B_t \\
d y_t \ne \dfrac{d B_t}{dt} \cdot dt \\
\implies y_t \ne e^{B_t}
$$

This is because $\dfrac{d B_t}{dt}$ is not defined since $B_t$ is not differentiable

## Ito’s Lemma

Consider $y_t = f(B_t)$, where $f$ is a smooth function
$$
\begin{aligned}
y_t &= f(B_t) \\
\implies df & \ne f'(B_t) \cdot d B_t \quad [\because (dB)^2 = dt] \\
\implies df &= f'(B_t) \cdot d B_t + \dfrac{1}{2} f''(B_t) \cdot dt
\end{aligned}
$$

## IDK

Assuming $\mu, \sigma$ are constant
$$
\begin{aligned}
dy_t &= \underbrace{\mu dt}_\text{Drift} + \sigma d B_t \\
\implies y_t &= \mu t + \sigma B_t
\end{aligned}
$$
Using Ito’s Lemma (Basically Taylor’s expansion)
$$
d f(t, x) = \dfrac{\partial f}{\partial t} + \mu \dfrac{\partial f}{\partial x} + \dfrac{1}{2} \sigma^2 \dfrac{\partial^2 f}{\partial x^2} + \dfrac{\partial f}{\partial x} d B_t
$$

## Integration

$$
\begin{aligned}
F(t, B_t) &= \int f(t, B_t) d B_t + \int g(t, B_t) dt \\
dF &= f dB_t + g dt
\end{aligned}
$$

Ito integral is the limit of Riemanian sums when we always take leftmost point of each integral

Intuitively, it only uses the data you have seen so far

## Adapted Process

A strategy/decision $D_t$ is said to be adapted to $y_t$, if $D_t$ only depends on $y_s, s \le t, \forall t$

If $D_t$ only depends on $t$ and not on $B_t$, then $y_t = \int D_t \cdot d B_t$ is normally-distributed at all times

## Ito Isometry

Used to calculate variance of Brownian motion
$$
\begin{aligned}
D_t &\text{ adapted to } B_t
\\
\implies
V(B_t)
&= E \left[ (\int_0^t D_s \cdot dB_s)^2 \right] \\
&= E \left[ \int_0^t D^2_t \cdot ds \right]
\end{aligned}
$$
Due to quadratic variance

## Martingale

If $g(t, B_t)$ is adapted to $B_t$ then $\int g(t, B_t) \cdot dB_t$, as long as $g$ is “reasonable”

$g$ is reasonable if $\int \int g^2 \cdot dt \cdot dB_t < \infty$

If a stochastic differential equation does not have a drift term, then it is a martingale
$$
d y_t = \sigma \cdot dB_t \qquad [\mu = 0]
$$
Defining stock price as brownian motion, as it is a martingale process
$$
\begin{aligned}
S_t &=
\exp(\frac{-\sigma^2 t}{2} + \sigma B_t) \\
\implies \dfrac{d S_t}{S_t} &= \sigma \cdot d B_t
\end{aligned}
$$

## Stochastic Differential Equation

$$
d y_t = \mu \cdot dt + \sigma \cdot dB_t
$$

## Change of measure

Consider 

- $B$ is brownian process w/ drift and pdf $P$
- $\tilde B$ is brownian process w/o drift and pdf $\tilde P$

$$
\exists z, \text{ such that } P(t) = z(t) \cdot \tilde P(t) \iff
P \equiv \tilde P
$$

$$
P \equiv P \text{ if } \\
P (A) > 0 \iff \tilde P (A) > 0,
\quad \forall A \subseteq \Omega
$$

 $z$ is called the Radon-Nikodym derivative

## Girsanov theorem

$$
z(t) = \dfrac{d \tilde P}{dP} (t) = e^{-\mu t T - \frac{\mu^2 T}{2}}
$$

$$
E[y_t] = \tilde E[\tilde z_t y_t] \\
\tilde E[y_t] = E[z_t y_t]
$$

