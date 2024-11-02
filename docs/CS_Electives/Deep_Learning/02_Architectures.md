# Architectures

|                                | Meaning                                                      | Efficient<br />at                      | Major<br />Application                                       | Computation<br />Complexity | Limitation                                                   | Advantage                                                    |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FC<br />Fully-Connected        |                                                              |                                        |                                                              |                             | Poor scalability for large input sizes<br />Do not capture “intuitive” invariances |                                                              |
| CNN<br />(Convolutional)       | - Require that activations between layers occur only in “local” manner<br />- Treat hidden layers themselves as spatial images<br />- Share weights across all spatial locations | Detecting spatial pattens              | Images, Videos                                               | High                        |                                                              | Reduce parameter count<br />Capture [some] “natural” invariances |
| RNN<br />(Recurrent)           | Forward-feed, backward-feed, and self-loop is allowed        | Detecting dependent/sequential pattens | Time Series                                                  |                             |                                                              |                                                              |
| ResNet<br />(Residual Network) |                                                              |                                        | Time Series                                                  |                             |                                                              |                                                              |
| U-Net                          |                                                              |                                        | Basis of diffusion models<br />Segmentation<br />Super-Resolution<br />Diffusion Models |                             |                                                              |                                                              |
| PINN<br />(Physics-Informed)   |                                                              |                                        |                                                              |                             |                                                              |                                                              |
| Lagrangian                     |                                                              |                                        |                                                              |                             |                                                              |                                                              |
| Deep Operator                  |                                                              |                                        |                                                              |                             |                                                              |                                                              |
| Fourier Neural Operator        |                                                              |                                        |                                                              |                             |                                                              |                                                              |
| Graph Neural Networks          |                                                              |                                        |                                                              |                             |                                                              |                                                              |