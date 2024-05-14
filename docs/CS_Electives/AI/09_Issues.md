# Issues

## Environmental Issues

Sources of carbon footprint

- train models
- data collection
- inference
- creating new chips

### Types

- Embodied carbon
- Operational carbon

Not just the energy cost of running computers, but also energy required to build them

![Untitled](./assets/Untitled.png)

### Solutions

- Efficiency of models
- Software infra to promote development of efficiency
  - AutoML instead of grid search
  - Easy-to-use APIs for efficient models
- Efficient computer chips
- Efficient datacenters
  - Microsoft: underwater servers
    - 8 x more reliable, using Nitrogen, etc
    - No need for fresh water for cooling
- Use renewable energy
  - Transitioning to 100% renewables will not eliminate the carbon footprint of chips
  - ![image-20240518102258790](./assets/image-20240518102258790.png)
- Stop planned obscelence

### 4 M’s of AI Efficiency

1. Model
2. Machine
3. Mechanism
4. Map: Location

### Jevon’s Paradox/Effect

Improved efficiency in resource utilization increases the total consumption of resources, due to increased rate of consumption from increased demand

Hence, definition of “efficiency” is important

### PUE

Power Usage Effectiveness

Lower is better

Typically $\in [1.1, 1.6]$

How the power input is being used for compute & for other supporting consumption such as cooling, lighting, etc.
$$
\begin{aligned}
\text{PUE}
&= \dfrac{\text{Total Energy}}{\text{Productive Energy}} \\
&=
1 + \dfrac{\text{Overhead}}{\text{Productive Energy}}
\end{aligned}
$$

## Ethical Issues

### Abstraction

Researchers/Engineers often abstracted away from application they’re working on, hence not aware of bad things that can be used for

### Data Ownership

Data governance

Trade-off: Privacy vs Progress

Model monetization trained on consumer data, eg: GitHub Copilot

### Data Bias

Dataset curation

Federated learning

## Research Inequality

- Inequitable access to computing resources
- Inequitable access to datasets