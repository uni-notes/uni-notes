## Flow Rates

$$
\begin{align}
\dot V &= vA \\\dot m &= \rho \dot V &= \rho vA \\&= \frac {\dot V}{\nu} &= \frac {vA}{\nu} 
&&
\begin{pmatrix}
\begin{align}
PV &= mRT \\ m &= PV \\ \rho &= \frac {P}{RT}
\end{align}
\end{pmatrix}
\end{align}
$$

### Flow Work

$W_\rm{f} = PV$

For non-flowing fluid (fluid that remains inside tank), Flow work = 0

## Conservation of Mass

$$
\begin{align}
\sum m_\rm{in} - \sum m_\rm{out} &= \Delta m \\\sum \dot m_\rm{in} - \sum \dot m_\rm{out} &= \frac {\mathrm{d} m}{\mathrm{d} t} \\\end{align}
$$

## Conservation of Energy

$$
\begin{align}
E_\rm{in} - E_\rm{out} &= \Delta E_\rm{cv} \\\dot E_\rm{in} - \dot E_\rm{out} &= \frac {\mathrm{d} E_\rm{cv}} {\mathrm{d} t} \\ \\
\dot Q_\rm{net} - \dot W_\rm{net} + \dot E_\rm{m, in} - \dot E_\rm{m, out} &= \frac {\mathrm{d} E_\rm{cv}} {\mathrm{d} t} \\ \\
\dot E_\rm{in}
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
\frac{\mathrm{d} m_\rm{cv}} {\mathrm{d} t} &= 0 \\ \\
\sum \dot m_\rm{in} &= \sum \dot m_\rm{out} \\\dot m_1 &= \dot m_2 \\ \\
\rho_1 v_1 A_1 &= \rho_2 v_2 A_2 \\\frac{v_1 A_1}{\nu_1} &= \frac{v_2 A_2}{\nu_2}

\end{align}
$$

### Energy

$$
\begin{align}
\frac{\mathrm{d} E_\text{cv}}{\mathrm{d} t} &= 0 \\ \\
\dot E_\rm{m, in} &= \dot E_\rm{m, out} \\\dot Q_\rm{net} - \dot W_\rm{net} + \dot E_\rm{m ,in} &= \dot E_\rm{m, out}
\end{align}
$$

## Steady Flow Devices

| Device                                | $v$  | $P$  | $T$  |                  Work                   |                                                              |
| ------------------------------------- | :--: | :--: | :--: | :-------------------------------------: | :----------------------------------------------------------: |
| Nozzle                                | inc  | dec  |      |                                         |                                                              |
| Diffuser                              | dec  | inc  |      |                                         |                                                              |
| Turbine<br />thermal $\to$ mechanical |      |      |      |          $\dot W_\rm{in} = 0$           |                                                              |
| Compressor                            |      | inc  | inc  |          $\dot W_\rm{out} = 0$          |                                                              |
| Throttling valve<br />(isenthalpic)   |      | dec  | dec  | $\dot W_\rm{in}  = \dot W_\rm{out} = 0$ | $\begin{align} h_1 &= h_2 \\ u_1 + P_1 \nu_1 &= u_2 + P_2 \nu_2 \end{align}$ |

## Unsteady/Transient Flow

### Mass

$$
\begin{align}
m_\rm{in} - m_\rm{out} &= \Delta m_\rm{cv} \\&= m_2 - m_1 \\
\dot m_\rm{in} - \dot m_\rm{out} &= \frac{\mathrm{d} m_\rm{cv}}{\mathrm{d} t}
\end{align}
$$

### Energy

$$
\begin{align}
E_\rm{in} - E_\rm{out} &= \Delta E_\rm{cv} \\Q_\rm{net} - W_\rm{net} + E_\rm{m, in} - E_\rm{m, out} &= \Delta E_\rm{cv} \\ \\
\Delta E_\rm{cv} &= m_2 e_2 - m_1 e_1 \\e &= h + \frac{v^2}{2000} + gz
\end{align}
$$

