# Gradients

## Gradient Issues

Especially for FP32 or lower precision

|                        | Gradients __ exponentially during back-propagation | Weight gradients | Cause                                      | Solution                                 |
| ---------------------- | -------------------------------------------------- | ---------------- | ------------------------------------------ | ---------------------------------------- |
| Vanishing (Converging) | shrink                                             | Too small        | Deep Networks                              | Weight-initialization<br>Weights Scaling |
| Exploding (Diverging)  | grow                                               | Too large        | Deep Networks                              | Weight-initialization<br>Clipping        |
|                        |                                                    |                  | Large loss due to target with large range* | Target normalization                     |

- A target variable with a large spread of values, in turn, may result in large error gradient values causing weight values to change dramatically, making the learning process unstable

## Gradient Clipping

rescales gradient to size at most $\theta$.

$$
g \leftarrow \min \left( 1, \frac{\theta}{\vert g \vert}  \right) g
$$

If the weights are large, the gradients grow exponentially during back-propagation
