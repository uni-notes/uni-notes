## Goal

- Sell a minimum number of seats without selling every seat at discount prices, such that it is enough to cover fixed operating costs
- Sell remaining seats at higher rates to maximize revenue

## Profit

$$
\begin{aligned}
\text{Profit}
&= \text{Income} - \text{Expenses} \\
&= \text{Sale Price} \times \min(\text{Demand}, \text{Quantity}) - \text{Cost} \times \text{Quantity}
\end{aligned}
$$

## Passengers

Passengers have different valuations

|                     | Business people | Others |
| ------------------- | --------------- | ------ |
| Keen on Flexibility | ✅               | ❌      |
| Booking Time        | Late            | Early  |
| Keen on refunds     | ✅               | ❌      |
| Price Elasticity    | Low             | High   |
| Purchasing Power    | High            | Low    |

## Selling Cases


|                                |                                             |
| ------------------------------ | ------------------------------------------- |
| Sell too many discounted seats | Not enough seats for high-paying passengers |
| Sell too many discounted seats | Empty seats at takeoff                      |

Lost revenue in both scenarios

## Optimization

We can formulate using [Optimization](../../Math_Electives/Optimization) 

- Objective Function: Maximize Total Revenue
- Constraints
  - Seats sold $>=$ 0
  - Seats sold $<=$ Capacity
  - Seats sold $<=$ Demand

## Shadow Price

Marginal revenue for unit increase in demand of regular seats
