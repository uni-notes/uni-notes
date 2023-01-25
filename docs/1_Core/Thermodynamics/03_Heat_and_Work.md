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
\begin{aligned}
F &= kx \\
W &= \frac{1}{2} k x^2 \\
&= \frac12 k ({x_2} ^2 - {x_1}^2) \\
\end{aligned}
$$

### Electric

$$
\begin{aligned}
\dot W &= VI \\
W &= VI \Delta t
\end{aligned}
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
| Isothermal | $\begin{aligned} T &= c \\ PV &= mRT \\ P_1 V_1 &= P_2 V_2 \end{aligned}$ | $P_i V_i \ \ln \vert  \frac{V_2}{V_1} \vert  \\ P_i V_i \ \ln \vert  \frac{P_1}{P_2} \vert$ | $mRT \ \ln \vert  \frac{V_2}{V_1} \vert$ <br /> $mRT \ \ln \vert  \frac{P_1}{P_2}  \vert$ |
| Polytropic | $\begin{aligned} P V^n &= c \\ P_1 (V_1)^n &= P_2 (V_2)^n \\ \frac{P_1}{P_2} &= \left( \frac{V_2}{V_1} \right)^n  \end{aligned}$ | $\frac{P_2 V_2 - P_1 V_1}{1-n}$ | $\frac{mR(T_2 - T_1)}{1-n}$ |

## Sign Convention

|    Quantity    | Sign |
| :------------: | :--: |
| $Q_\text{in}$  |  +   |
| $Q_\text{out}$ |    -   |
| $W_\text{in}$  |    -   |
| $W_\text{out}$ |  +   |
|   expansion    |  +   |
|  compression   |    -   |

