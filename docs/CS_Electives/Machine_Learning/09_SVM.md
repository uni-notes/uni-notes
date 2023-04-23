## Support Vector Machine

Goal is to obtain hyperplane farthest from all sample points

$$
\begin{aligned}
\text{Distance } & \text{between edge point and line} \\
&= \frac{|w^t x_i + w_0|}{||w||} \\
&=\frac{1}{||w||} \\
\implies m &= \frac{2}{||w||}
\end{aligned}
$$

Goal is to maximize ‘margin’ $m$ (distance between classes), subject to the following constraints

$$
\begin{cases}
w^t x_i + w_0 \ge 1, & x_i > 0 \\
w^t x_i + w_0 \le -1, & x_i <0 
\end{cases}
$$

In other words, we need to minimize cost function

$$
J(\theta) = \frac{1}{2} ||w||^2
$$

We can derive through linear-programming

## For Linearly-Separable

1. Plot sample points

2. Find support vectors (points that are on border of other class)

3. Find augmented vectors with bias = 1
   
$$
s_1 = \begin{pmatrix} 0 \\
 1 \end{pmatrix}
\implies
\tilde{s_1} = \begin{pmatrix} 0 \\
 1 \end{pmatrix}
$$

4. Find values of $\alpha$, assuming that

     - $+ve = +1$
     - $-ve = -1$

$$
\begin{aligned}
\alpha_1 \tilde{s_1} \cdot \tilde{s_1} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_1} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_1}
&= -1 \\
\alpha_1 \tilde{s_1} \cdot \tilde{s_2} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_2} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_2}
&= 1 \\
\alpha_1 \tilde{s_1} \cdot \tilde{s_3} +
\alpha_2 \tilde{s_2} \cdot \tilde{s_3} +
\alpha_3 \tilde{s_3} \cdot \tilde{s_3}
&= 1
\end{aligned}
$$

5. Find $w_i$
   
$$
w_i =
$$

6. Something

## Kernel function $\phi(x)$ 

Linear transformation function for Non-Linearly-Separable

For eg, to increase the dimensionality, we can use $\phi(x) = (x, x^2)$

| Kernel Function                                              | $\phi(x)$                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Linear                                                       | $x$                                                          |
| Polynomial                                                   | $(kx+c)^n$                                                   |
| Gaussian                                                     | $\exp \left( \dfrac{-\| x-y \|^2}{2 \sigma^2} \right)$ <br /> where $\sigma^2 =$ Variance of sample|
| RBF<br />(Radial Basis Function)<br />Most powerful, but not necessary in most cases | $\exp( -\gamma \| x_i - x_j \|^2 )$                            |

