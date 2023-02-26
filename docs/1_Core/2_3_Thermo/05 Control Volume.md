## Flow Rates

$$
\begin{align}
\dot V &= vA \\
\dot m &= \rho \dot V = \rho vA \\
&= \frac{\dot V}{\nu} = \frac{vA}{\nu} \\
\Big( PV &= mRT, m = PV, \rho = \frac{P}{RT} \Big)
\end{align}
$$

### Flow Work

$W_\text{f} = PV$

For non-flowing fluid (fluid that remains inside tank), Flow work = 0

## Conservation of Mass

$$
\begin{align}
\sum m_\text{in} - \sum m_\text{out} &= \Delta m \\
\sum \dot m_\text{in} - \sum \dot m_\text{out} &= \frac{\mathrm{d} m}{\mathrm{d} t} \\
\end{align}
$$

## Conservation of Energy

$$
\begin{align}
E_\text{in} - E_\text{out} &= \Delta E_\text{cv} \\
\dot E_\text{in} - \dot E_\text{out} &= \frac{\mathrm{d} E_\text{cv}}{\mathrm{d} t} \\
\dot Q_\text{net} - \dot W_\text{net} + \dot E_\text{m, in} - \dot E_\text{m, out} &= \frac{\mathrm{d} E_\text{cv}}{\mathrm{d} t} \\
\dot E_\text{in}
&= \dot m \left[ h + \frac{v^2}{2000} + gz \right]
& (h = u + P\nu) \\
&= \dot m \left[ u + \frac{v^2}{2000} + gz \right] 
& \text{(non-flowing)}
\end{align}
$$

## Steady Flow

Properties within the control volume remain constant with time

### Mass

$$
\begin{align}
\frac{\mathrm{d} m_\text{cv}}{\mathrm{d} t} &= 0 \\
\sum \dot m_\text{in} &= \sum \dot m_\text{out} \\
\dot m_1 &= \dot m_2 \\
\rho_1 v_1 A_1 &= \rho_2 v_2 A_2 \\
\frac{v_1 A_1}{\nu_1} &= \frac{v_2 A_2}{\nu_2}
\end{align}
$$

### Energy

$$
\begin{align}
\frac{\mathrm{d} E_\text{cv}}{\mathrm{d} t} &= 0 \\
\dot E_\text{m, in} &= \dot E_\text{m, out} \\
\dot Q_\text{net} - \dot W_\text{net} + \dot E_\text{m ,in} &= \dot E_\text{m, out}
\end{align}
$$

## Steady Flow Devices

| Device                                | $v$  | $P$  | $T$  |                    Work                     |                                                              |
| ------------------------------------- | :--: | :--: | :--: | :-----------------------------------------: | :----------------------------------------------------------: |
| Nozzle                                | inc  | dec  |      |                                             |                                                              |
| Diffuser                              | dec  | inc  |      |                                             |                                                              |
| Turbine<br />thermal $\to$ mechanical |      |      |      |           $\dot W_\text{in} = 0$            |                                                              |
| Compressor                            |      | inc  | inc  |           $\dot W_\text{out} = 0$           |                                                              |
| Throttling valve<br />(isenthalpic)   |      | dec  | dec  | $\dot W_\text{in}  = \dot W_\text{out} = 0$ | $\begin{align} h_1 &= h_2 \\ u_1 + P_1 \nu_1 &= u_2 + P_2 \nu_2 \end{align}$ |

## Unsteady/Transient Flow

### Mass

$$
\begin{align}
m_\text{in} - m_\text{out} &= \Delta m_\text{cv} \\&= m_2 - m_1 \\
\dot m_\text{in} - \dot m_\text{out} &= \frac{\mathrm{d} m_\text{cv}}{\mathrm{d} t}
\end{align}
$$

### Energy

$$
\begin{align}
E_\text{in} - E_\text{out} &= \Delta E_\text{cv} \\
Q_\text{net} - W_\text{net} + E_\text{m, in} - E_\text{m, out} &= \Delta E_\text{cv} \\
\Delta E_\text{cv} &= m_2 e_2 - m_1 e_1 \\
e &= h + \frac{v^2}{2000} + gz
\end{align}
$$
