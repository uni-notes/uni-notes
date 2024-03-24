# Hyper-Parameter Tuning

|                |                                                              | Advantage | Disadvantage              |
| -------------- | ------------------------------------------------------------ | --------- | ------------------------- |
| Manual         |                                                              |           | Time-Consuming            |
| Grid Search    |                                                              |           | Computationally-expensive |
| Random Search  |                                                              |           | Non-deterministic         |
| Evolutionary   | Randomization, Natural Selection, Mutation                   |           |                           |
| Bayesian       | Probabilistic model of relationship b/w cost function and hyper-parameters, using information gathered from trials |           |                           |
| Gradient-Based | Treat hyper parameter tuning like parameter fitting          |           |                           |
| Early-Stopping | Focus resources on settings that look promising<br />eg: Successive Halving |           |                           |

## Speed Up

- Add more processors
- Caching
- Random sampling: Won’t work with caching

![image-20240317160544276](./assets/image-20240317160544276.png)