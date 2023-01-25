## Terms

| Term        | Meaning                    |                    Formula                     |
| ----------- | -------------------------- | :--------------------------------------------: |
| $\eta$      | Efficiency                 | $\frac{\text{Desired Output}}{\text{Input}}$ |
| COP         | Coefficient of Performance | $\frac{\text{Desired Output}}{\text{Input}}$ |
| $q$         | Calorific/Heating Value    |                  $\frac Q m$                   |
| Gravimetric | mass terms                 |                                                |

## Devices

| Device        | Purpose                                |                                                              |                                                              |                                                              |
| ------------- | -------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Heat Engine   | - Heat $\to$ Work<br />- cycle         | $\begin{aligned} \eta_{\small\text{HE}} &= \frac{W_\text{net, out}}{Q_\text{H}} \\
 &= 1 - \frac{Q_\text{L}}{Q_\text{H}} \end{aligned}$ |        $\eta_\text{HE} < 1$<br />Kelvin-Plank Statement        | $\begin{aligned} \Delta U &= 0 \\ Q_\text{net} &= W_\text{net} \\ W_\text{net, out} &= Q_\text{in} - Q_\text{out} \\ &= Q_\text{H} - Q_\text{L} \end{aligned}$ |
| Refridgerator | - maintain cool temp<br />- Reverse HE | $\begin{aligned} \text{COP}_R &= \frac{Q_\text{L}}{Q_\text{net, in}} \\
 &= \frac{1}{ \frac{Q_\text{H}}{Q_\text{L}} - 1 } \end{aligned}$ |                   $\text{COP}_R$ can be > 1                    |                                                              |
| Heat Pump     | - maintain warm temp<br />- Reverse HE | $\begin{aligned} \text{COP}_{HP} &= \frac{Q_\text{H}}{W_\text{net, in}} \\
 &= \frac{1}{ 1 - \frac{Q_\text{L}}{Q_\text{H}} } \end{aligned}$ | $\begin{aligned} \text{COP}_{HP} &= \text{COP}_{R} + 1 \\ \text{COP}_{HP} &> \text{COP}_{R} \end{aligned}$ |                                                              |

```mermaid
flowchart LR

subgraph Heat Engine
direction LR
a([Warm]) -->
|Q<sub>H</sub>| b[System] -->
|Q<sub>L</sub>| c([Cool])

b --> |W<sub>net</sub>| d[ ]
end

subgraph Refridgerator/Heat Pump
direction LR
r([Cool]) -->
|Q<sub>L</sub>| q[System] -->
|Q<sub>H</sub>| p([Warm])
s[ ] --> |W<sub>net</sub>| q
end
```

## Carnot Cycle

For Heat Engine

Adiabatic means polytropic process with**out** heat transfer

| Transition | Characteristic                            |    Constant     |            Signs             |                             Work                             |
| :--------: | ----------------------------------------- | :-------------: | :--------------------------: | :----------------------------------------------------------: |
|   1 - 2    | Isothermal Expansion<br />Heat Absorbed   |    $PV = c$     | $W_{12} > 0 \\
 Q_\text{H} > 0$ | $P_1 V_1 \ln \| \frac{V_2}{V_1} \|$ <br /> $P_2 V_2 \ln \| \frac{P_1}{P_2} \|$ |
|   2 - 3    | Adiabatic Expansion                       | $PV^\gamma = c$ |         $W_{23} > 0$         |               $\frac{P_3 V_3 - P_2 V_2}{1-n}$                |
|   3 - 4    | Isothermal Compression<br />Heat Released |    $PV = c$     | $W_{34} < 0 \\
 Q_\text{L} < 0$ | $P_3 V_3 \ln \| \frac{V_4}{V_3} \|$ <br /> $P_4 V_4 \ln \| \frac{P_3}{P_4} \|$ |
|   4 - 1    | Adiabatic Compression                     | $PV^\gamma = c$ |         $W_{41} < 0$         |               $\frac{P_1 V_1 - P_4 V_4}{1-n}$                |

$$
\begin{aligned}
W_\text{net, out} &= W_{12} + W_{23} + W_{34} + W_{41} \\
\eta
&= \frac{W_\text{net, out}}{Q_\text{H}} \\
&= 1 - \frac{Q_\text{L}}{Q_\text{H}} \\
&= 1 - \frac{T_L}{T_H}
\end{aligned}
$$

Make sure of the signs when calculating $W_\text{net, out}$

## Reverse Carnot Cycle

For Refridgerator, Heat Pump

$Q_\text{L} > 0, Q_\text{H} < 0$

$$
\begin{aligned}
W_\text{net, in} &= W_{12} + W_{23} + W_{34} + W_{41} \\
\text{COP}_R &= \frac{Q_\text{L}}{W_\text{net, in}} \\
\text{COP}_{HP} &= \frac{Q_\text{H}}{W_\text{net, in}}
\end{aligned}
$$

