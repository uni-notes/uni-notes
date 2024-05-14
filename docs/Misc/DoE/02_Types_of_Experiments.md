# Types of Experiments

| Type                                 |                                                              | Example                                                      |                                                              |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Natural/Quasi                        | In non-experimental settings, sometimes implicit randomization occurs, and the treatment occurs “as if” it is random | Uni admission cutoff provides a natural experiment on uni education. Students **just** above/below are likely to be very similar. For these students, uni education is “as if” random. Comparing these students (ones that went to uni/not) produces an estimate of the causal effect of college education. |                                                              |
| Regression Discontinuity Design      | Discrete treatment status determined by an underlying continuous variable, which is used for quasi experiments |                                                              | ![image-20240213172957152](./assets/image-20240213172957152.png) |
| Differences-in-Differences           | 2 time-series process $y_1$ and $y_2$ have the factors affecting them |                                                              | ![image-20240213175121148](./assets/image-20240213175121148.png) |
| RCT<br />(Randomized Control Trials) |                                                              |                                                              |                                                              |

### Types of RDD

|       | $x$                                                          |
| ----- | ------------------------------------------------------------ |
| Sharp | $\begin{cases} 1, & z \ge z_0\\ 0, & \text{o.w}\end{cases}$  |
| Fuzzy | $\begin{cases} p(z), & z \ge z_0\\ 0, & \text{o.w}\end{cases}$ |
