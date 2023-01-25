## Derivatives

$$
\begin{align}
L\{ f'(t) \} &= s F(s) - f(0) \\L\{ f''(t) \} &= s^2 F(s) - sf(0) - f'(0) \\\Big( L\{ f(t) \} &= F(s) \Big)
\end{align}
$$

$$
\begin{align}
f(0) &= \{ f(t) \}_{t = 0} \\f'(0) &= \left\{ \frac{d f(t)}{dt} \right\}_{t = 0} \\f'(t) &= \frac{df}{dx}; f''(t) = \frac{d^2f}{dx^2}
\end{align}
$$

### I missed after this

$$
\begin{align}
L^{-1} \left[ \frac{F(s)}{s} \right]
&= \int\limits_0^t L^{-1} \Big( F(s) \Big)  \ dt\\
L^{-1} \left[ \frac{F(s)}{s^2} \right]
&= \int\limits_0^t \int\limits_0^t L^{-1} \Big( F(s) \Big)  \ dt\\
L^{-1} \left[ \frac{F(s)}{s^n} \right]
&= \text{n integrals from } 0 \to t \quad L^{-1} \Big( F(s) \Big)  \ dt
\end{align}
$$

