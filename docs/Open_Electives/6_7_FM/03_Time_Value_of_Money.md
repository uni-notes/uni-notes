## Time Value of Money

|                                                              | Denotion   | Expressed as | Value of something at                                 |
| ------------------------------------------------------------ | ---------- | ------------ | ----------------------------------------------------- |
| Present Value                                                | PV         | Currency     | $t = 0$ (not even $t \approx 0$)                      |
| Future Value                                                 | FV         | Currency     | $t > 0$                                               |
| Interest Rate<br />Discount Rate<br />Compound Rate<br />Opportunity cost of capital<br />Required return | $r$        | %            | Exchange rate between present & future value          |
| Number of Periods                                            | $n$ or $t$ |              |                                                       |
| Timeline                                                     |            |              | Graphical reprsesentation of the timing of cash flows |

You should never compare money across different time instants. We can only compare at the same instant.

| When we take cashflow ___ in time | Name        |
| --------------------------------- | ----------- |
| forward                           | compounding |
| backward                          | discounting |

Return for every investment is a compensation

- Time
- Inflation
- Risk

$$
\begin{align}
\text{FV} &= \text{PV} \times \underbrace{(1+r)^t}_{\text{Compound Factor}} \\
\implies \text{PV} &= \text{FV} \times \underbrace{\frac{1}{(1+r)^t}}_\text{Discount Factor}
\end{align}
$$

In a finance interview, if you’re not sure of the answer, just say it’s compounding 😭😂

## Multiple Cashflows


$$
\begin{align}
\text{FV} &= \sum_{i=1}^n C_i (1+r)^i \\
\text{PV} &= \sum_{i=1}^n \frac{C_i}{(1+r)^i}
\end{align}
$$

## Types of Interest

If $P =$ original principal amount

| Type                    | FV                        |
| ----------------------- | ------------------------- |
| Simple                  | $P \times (1+r) \times t$ |
| Compound<br />(Default) | $P \times (1+r)^t$        |
