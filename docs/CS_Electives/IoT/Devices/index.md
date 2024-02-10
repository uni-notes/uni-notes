# Devices

## Sensors (Input)

Devices that detects the state of a physical environment, and quantitatively provides a corresponding output as an electrical/optical signal.

### Sensor Fusion

Combining measurements of the same quantity from multiple sensors, to obtain a combined information with lower uncertainty than any of the individual sensors. Using multiple sensors for the quantity also allows us to verify each sensor wrt others.

If we have $s$ sensors,
$$
\begin{aligned}
\mu_\text{S} &=
\left( \sum \limits_s^S \dfrac{\mu_s}{\sigma^2_s} \right) \sigma^2_{S} \\
\sigma^2_\text{S} &=
\dfrac{1}{\sum \limits_s^S \dfrac{1}{\sigma^2_s} }
\end{aligned}
$$
where $S$ refers to the combination of all the sensors

## Effectors (Output)

Devices that perform some action such as emitting light, sound, motor, etc
