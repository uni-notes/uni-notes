## Addable Quantities

- mass
- volume
- U
- H
- $u$ for closed system

note that specific quanties like $h, u$ can ***not*** be added

## Work

### Spring

$$
\begin{align}
F &= kx \\
W &= \frac{1}{2} k x^2 \\
&= \frac12 k ({x_2} ^2 - {x_1}^2) \\
\end{align}
$$

### Electric

$$
\begin{align}
\dot W &= VI \\
W &= VI \Delta t
\end{align}
$$

### Boundary Work

Note that temperature should be in $K$ (Kelvin)

$$
W_\text{out, b} = \int \limits_{v_1}^{v_2} P \cdot dv
$$

|    Type    |                         Condition(s)                         |                            $W_b$                             |                                                              |
| :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Isochoric  |                           $V = c$                            |                             $0$                              |                                                              |
|  Isobaric  |                           $P = c$                            |                       $P_1(V_2 - V_1)$                       |                    $mP_1(\nu_2 - \nu_1)$                     |
| Isothermal | $\begin{align} T &= c \\ PV &= mRT \\ P_1 V_1 &= P_2 V_2 \end{align}$ | $P_i V_i \ \ln \| \frac{V_2}{V_1} \| \\ P_i V_i \ \ln \| \frac{P_1}{P_2} \|$ | $mRT \ \ln \| \frac{V_2}{V_1} \|$ <br /> $mRT \ \ln \| \frac{P_1}{P_2} \|$ |
| Polytropic | $\begin{align} P V^n &= c \\ P_1 (V_1)^n &= P_2 (V_2)^n \\ \frac{P_1}{P_2} &= \left( \frac{V_2}{V_1} \right)^n  \end{align}$ | $\frac{P_2 V_2 - P_1 V_1}{1-n}$ | $\frac{mR(T_2 - T_1)}{1-n}$ |

## Sign Convention

|    Quantity    | Sign |
| :------------: | :--: |
| $Q_\text{in}$  |  +   |
| $Q_\text{out}$ |    -   |
| $W_\text{in}$  |    -   |
| $W_\text{out}$ |  +   |
|   expansion    |  +   |
|  compression   |    -   |

