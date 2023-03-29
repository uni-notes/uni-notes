## Cointegrating Processes

2 processes that are integrated of order 1, but $\exists$ linear combination of the 2 variables that is stationary.

Tendancy of 2 variables (that are theoretically equilibrium) to be related to each other

If there is divergence, it will only be temporary, as there is bound to be **error correction**

The coefficient associated with the 2 variables will be **non**-zero

Usually happens with highly connected variables

If there are $n$ cointegrating variables, then there can be

- $[1, n-1]$ **independent** cointegrating relationships (not lesser or greater than this range)
- $[1, n]$ error correction relationships

eg:

- Demand and Supply for a commodity
- US interest rate and UAE interest rate
    - US is leading market
    - UAE is following market
- Dubai and Sharjah rent
- GCC stock markets

Consider $x, z$ which are both $I(1)$ processes; $x_t$ and $z_t$ are cointegrated processes $\iff u_t$ is stationary process,

$$
\begin{aligned}
z_t &= \alpha_1 x + u_t & \text{(Long-Term Specification)} \\
\implies u_t &= z_t - \alpha_1 x_t & \text{(Short-Term Specification)} \\
z_t - z_{t-1} &= \textcolor{hotpink}{-}\alpha_D(z_{t-1} - \alpha_1 x_{t-1}) + v_t \\
\Delta z_t &= \textcolor{hotpink}{-}\alpha_D(u_{t-1}) + v_t \\
& \text{if } x \text{ also has correcting tendancy,} \\
\implies \Delta x_t &= \textcolor{orange}{+} \alpha_G(u_{t-1}) + w_t
\end{aligned}
$$

- $\alpha_D$
    - Speed of adjustment parameter, or error correction coefficient
    - $\alpha_D \in (0, 1)$

### Parts

- Attractor/Leader
- Attracted/Follower

## Drunk Couple and Dog

## Error Correction Models

$$
\begin{aligned}
\Delta m_t
&= \lambda_m (u_{t-1}) + \epsilon_{mt} \\&= \lambda_m (blah blah) + \epsilon_{mt} \\
\Delta p_t &= \\
\Delta y_t &= 
\end{aligned}
$$

- $\epsilon$ is white noise error
- $\lambda$ are velocity of adjustment parameters

Atleast one of the $\lambda$ should be significant, otherwise there is no error correction $\implies$ no cointegration

> Cointegration and error correction are equivalent representation
>
> ~ Granger representation theorem

## Vector regression/Structural estimation model

Used when there is no cointegration

I missed this

If the values of $\lambda$ are zero, then it is a simple VAR model and there is no cointegration. 

![image-20221226212830201](assets/image-20221226212830201.png){ loading=lazy }

## Example: Quantity Theory of Money

$$
MV = \underbrace{PY}_{\text{GDP}}
$$

|      |                         |                                                             | Integrated of order       |
| ---- | ----------------------- | ----------------------------------------------------------- | ------------------------- |
| $M$  | Total quantity of money |                                                             | $I(1)$                    |
| $V$  | Velocity of money       | Number of times a unit of currency is transferred in a year | N/A<br />(Constant value) |
| $P$  | Price                   |                                                             | $I(1)$                    |
| $Y$  | Real quantity of Output |                                                             | $I(1)$                    |

As they are $I(1)$, they are not mean-reverting variables. Hence, taking log on both sides of equation, and then transposing

$$
\beta_0 + \beta_1 m_t - \beta_2 p_t - \beta_3 y_t = u_t
$$

Velocity is a constant, which is an intercept. Here it is represented by $\beta_0$, but can also represented by $1\cdot V$

If $u_t$ is $I(0) \implies M, V, P, Y$ are cointegrating

### Notes

1. There can be multiple cointegrating vectors $\{\beta_0, \beta_1, \beta_2, \beta_3 \} = \{\lambda \beta_0, \lambda \beta_1, \lambda \beta_2, \lambda \beta_3 \} \iff \lambda \ne 0$
2. If $m$ and $p$ are $I(2)$ whereas $y$ is $I(1)$. The linear combination of these three variables will be $I(2)$, hence the 3 are not cointegrated
3. However, if a linear combination $\beta_1 m + \beta_2 p$ is $I(1)$, and this is cointegrated with y which is $I(1)$, then we say there is multi-cointegration
4. if monetary policy folows feedback rule that changes money supply based on inflation, then inflation will be another cointegrated variable

## Granger Causality

Let’s say we have 2 variables $x, y$. We can check if $x$ granger causes $y$

$$
y_t = \beta_1 y_{t-1} + \beta_2 x_{t-1} + u_t
$$

### Hypotheses

- $H_0: \beta_2 = 0$ 
    - $y$ is independent of $x$
    - $x$ does not granger cause $y$
- $H_1: \beta_2 \ne 0$
    - $x \to y$
    - $x$ granger causes $y$

### Procedure

1. We check if the $R_{adj}^2$ has increased by incorporating $x_{t-1}$, when compared to without it $(y_t = \beta_1 y_{t-1} + u_t)$
2. Do a hypothesis test
3. If $p \le 0.05,$ reject null hypothesis, and hence conclude that $x \to y$
