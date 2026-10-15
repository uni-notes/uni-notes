These are network-based methods, designed to assist in planning, scheduling, and control of projects.

## Objective

1. Determine the minimum possible completion time for the project.
2. Determine a range of start and end time for each activity, so that the project can be completed in minimum time.

## Notations

|                             |                                          |
| :-------------------------: | ---------------------------------------- |
|  $\text{ES}_j = \square_j$  | Earliest occurance time of event $j$     |
| $\text{LC}_j = \triangle_j$ | Latest completion time of event $j$      |
|          $D_{ij}$           | Duration of activity between $i$ and $j$ |

## Activities

|                                  | Critical                                                     | Non-Critical         |
| -------------------------------- | ------------------------------------------------------------ | -------------------- |
| Leeway in determining start time | ❌                                                            | ✅, within some limit |
| Leeway in determining end time   | ❌                                                            | ❌                    |
| Conditions                       | 1. $\square_i = \triangle_i$<br />2. $\square_j = \triangle_j$<br />3. $\square_j - \square_i = \triangle_j - \triangle_i = D_{ij}$ |                      |

where $D_{ij}$ =  given distance between 2 nodes

## Methods

|                         | CPM                                                          | PERT                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Full Form               | Critical Path Method                                         | Project Evaluation Review Technique                          |
| Assumption for activity | deterministic durations                                      | probabilistic durations                                      |
| Duration of activity    | Fixed                                                        | Determined based on<br />- most optimistic time $a$<br />- most likely time $m$<br />- pessimistic time $b$ |
| Procedure               | 2 passes<br />1. Forward pass determines earliest occurance times; take path with max duration if $\exists$ multiple paths<br />2. Backward pass determines latest completion times; take path with min duration if $\exists$ multiple paths<br />3. Find critical paths<br />4. Find the [float](#float) for non-critical activities | 1. Calculate distance and variance<br />2. Solve like CPM<br />3. Calculate cumulative E(D_i) and Var<br />4. Calculate required probabilities using $z$ distribution<br />In case of ties, take the max variance path, thereby reflecting more uncertainty |
|                         |                                                              | Average duration $\bar D = \frac{a+4m+b}{6}$<br />Variance $= \left(\frac{b-a}{6}\right)^2$ |

## Float

| Free Float                       | Total Float                        |
| -------------------------------- | ---------------------------------- |
| $\square_j - \square_i - D_{ij}$ | $\triangle_j - \square_i - D_{ij}$ |

| Case    |                                                              |
| ------- | ------------------------------------------------------------ |
| FF = 0  | Any delay will cause delay in starting successive activities |
| FF < TF | We have leeway in starting the project as FF units<br />For any excess delay (FF < d < T), starting successive activities will be delayed |
| FF = TF | Activities may be scheduled anywhere between the earliest start time & the latest completion time without delaying the project |
