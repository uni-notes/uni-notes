# Automatic Differentiation

## Differentiation Types

|                        |                                                              | Error           | Disadvantage                                                 |
| ---------------------- | ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ |
| Numerical              | $\lim \limits_{\epsilon \to 0} \dfrac{f(x + \epsilon) - f(x)}{\epsilon}$ | $O(\epsilon)$   | Less accurate                                                |
| Numerical<br />Type 2  | $\lim \limits_{\epsilon \to 0} \dfrac{f(x + \epsilon) - f(x-\epsilon)}{2\epsilon}$ | $O(\epsilon^2)$ | Numerical error<br />Computationally-expensive               |
| Symbolic               | Derive gradient by sum, product, chain rules                 |                 | Tedious<br />Computationally-expensive                       |
| Backprop               | Run backward operations the same forward graph               |                 |                                                              |
| Forward mode automatic | Output: Computational graph<br /><br /><br />Define $\dot v_i = \dfrac{\partial v_i}{\partial x_j}$<br />where $v_i$ is an intermediate result |                 | Computationally-expensive: $n$ forward passes required to get gradient of each input |
| Reverse mode automatic | Output: Computational graph<br /><br />Define adjoint $\bar v_i = \dfrac{\partial y}{\partial v_i}$<br />where $v_i$ is an  intermediate result<br />$\overline{v_{k \to i}} = \bar v_i \dfrac{\partial v_i}{\partial v_k}$ |                 |                                                              |

### Numerical gradient checking

$$
\Delta^T \nabla_x f(x) = \dfrac{f(x + \epsilon \delta) - f(x - \epsilon \delta)}{2 \epsilon} + O(\epsilon^2)
$$

Pick $\delta$ from unit ball