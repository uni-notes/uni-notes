## Complex Numbers

$$
z = x + iy
$$

==Make sure that all calculations are in radian==

## Properties

$$
\begin{aligned}
|z| &= \sqrt{x^2 + y^2} \\
|z_1 \cdot z_2| &= |z_1| \cdot |z_2| \\
\left| \frac{z_1}{z_2} \right| &= \frac{ |z_1| }{ |z_2| } \\
\bar z &= x - iy \\
|\bar z| &= |z| \\
\bar{ |z| }^2 &= z \cdot \bar z \\
\frac{z + \bar z}{2} &= \text{Re}(z) \\
\frac{z - \bar z}{2i} &= \text{Im}(z) \\
\overline{z_1 \pm z_2} &= \bar z_1 \pm \bar z_2 \\
\overline{z_1 \cdot z_2} &= \bar z_1 \cdot \bar z_2 \\
\overline{\left( \frac{z_1}{z_2} \right)} &= \frac{\bar z_1}{\bar z_2}
\end{aligned}
$$

### Circles

|                   |                                   |
| ----------------- | --------------------------------- |
| $\| z \| = r$     | circle with radius $r$ @ $(0, 0)$ |
| $\| z-z_0 \| = r$ | circle with radius $r$ @ $z_0$    |

## Triangle Inequality

| Upper Bound                         | Lower Bound                                    |
| ----------------------------------- | ---------------------------------------------- |
| $\| z_1 \pm z_2 \| \le \| z_1 \| + \| z_2 \|$ | $\| z_1 \pm z_2 \| \ge \text{abs} (\| z_1 \| - \| z_2 \|)$ |

**abs** refers to absolute value

## Argument

$$
\begin{aligned}
\text{arg } z &= \left| \frac{y}{x} \right| \\
\text{Arg } z &= \text{Principle Value of arg } z\\
\text{arg}(z_1 \cdot z_2) &= \text{arg}(z_1) + \text{arg}(z_2) \\
\text{arg}\left( \frac{z_1}{z_2} \right) &= {\text{arg}(z_1)} - {\text{arg}(z_2)}
\end{aligned}
$$

## Polar Form

$$
\begin{aligned}
z
&= r \cdot e^{i \theta} \\
&= r (\cos \theta + i \sin \theta)
\end{aligned}
$$

## Root

$$
\begin{aligned}
c
&= (r \cdot e^{i\theta})^{\frac{1}{n}} \\
&= r^{\frac{1}{n}} \cdot e^{\frac{i\theta}{n}} \\
&= r^{\frac{1}{n}} \Bigg(
	\cos \left(\frac{\theta}{n}\right) + i \sin \left(\frac{\theta}{n}\right)
\Bigg) \\
r &= |z| \\
\frac{\theta}{n} &= \frac{\text{Arg }z + 2k\pi}{n}, k \in [0, n) \\
e^{i(n\theta)} &= \cos(n\theta) + i \sin(n\theta) \\
e^{-i(n\theta)} &= \cos(n\theta) - i \sin(n\theta)
\end{aligned}
$$

