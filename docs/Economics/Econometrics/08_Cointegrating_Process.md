# Cointegrating Processes

Tendency of 2 variables (that are theoretically at equilibrium) to be related to each other

2 processes that are integrated of order 1, but $\exists$ linear combination of the 2 variables that is stationary.

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
- $u_t=$ Disequilibrium error/Cointegration residual

### Parts

- Attractor/Leader
- Attracted/Follower

## Correlation vs Co-integration

Co-integration  $\  \not \!\!\!\!\! \iff$ Correlation

|                           | Correlation | Co-Integration |
| ------------------------- | ----------- | -------------- |
| Co-movement<br />Duration | short-term  | long-term      |

## Drunk Couple and Dog
