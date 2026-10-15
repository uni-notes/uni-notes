blah

## Properties

| Condition    |                   Meaning                   | Mathematical representation                                  |
| ------------ | :-----------------------------------------: | ------------------------------------------------------------ |
| Capacity     | $0 \le \text{flow}_e \le \text{capacity}_e$ | $0 \le f(e) \le c_e, \ \forall e \in E$                      |
| Conservation |        Inflow = Outflow @ every node        | $\sum\limits_\text{Inflow} f(e) = \sum\limits_\text{Outflow} f(e), \ \forall e \in E$ |

## Algorithm

1. Find all paths from source to destination
2. The maximum flow is limited by the bottleneck, which is the lowest value in a path, ie $\text{argmin} (c_e), \ e \in P$

## Residual Graph

Indicates how much more flow is allowed in each edge in the network graph

| Path            | Direction of flow in residual graph            |
| --------------- | ---------------------------------------------- |
| Unused          | Same                                           |
| Partially used  | Used will be reversed<br />Unused will be same |
| Completely used | Reversed                                       |

