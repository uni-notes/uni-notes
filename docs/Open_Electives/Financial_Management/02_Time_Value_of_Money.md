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
\text{FV} &= \sum_{t=1} c_t (1+r)^t \\
\text{PV} &= \sum_{t=1} \frac{c_t}{(1+r)^t}
\end{align}
$$

If one of the cashflow happens in first year itself, then it will be

$$
\text{PV} = c_0 + \sum_{t=1} \frac{c_t}{(1+r)^t}
$$


## Types of Interest

If $P =$ original principal amount

| Type                    | FV                        |
| ----------------------- | ------------------------- |
| Simple                  | $P \times (1+r) \times t$ |
| Compound<br />(Default) | $P \times (1+r)^t$        |

## Types of Cashflows

Infinite series of cashflows which has

eg: Preference share in a corporation

|                          | Perpetuity    | Annuity                                          |
| ------------------------ | ------------- | ------------------------------------------------ |
| Finiteness               | Infinite      | Finite                                           |
| Term                     | Forever       |                                                  |
| Fixed Cashflow           | ✅             | ✅                                                |
| Occurs every time period | ✅             | ✅                                                |
| Present Value            | $\frac{c}{r}$ | $\frac{c}{r} \left[ 1-\frac{1}{(1+r)^t} \right]$ |
| Future Value             | N/A           | $\frac{c}{r} \left[ (1+r)^t - 1 \right]$         |

### Conceptual understanding of long-term loan

Every [equal] installment is actually a combination of

- interest payment
- principal repayment

As time goes on, your installment will be constituting: less of interest repayment & more of principal repayment

## I missed a few classes

## Interest Rates

### APR

Annual Percentage Rate

### EAR

Effective Annual Rate

The actual interest rate you are paying

$$
\text{EAR } = \left(
1 + \frac{\text{APR}}{m}
\right)^m - 1
$$

where $m =$ interest compounding frequency

This is the value of $r$ we use when calculating present/future value

### Compounding Frequency

|             | $m$                |
| ----------- | ------------------ |
| Annual      | 1                  |
| Semi-Annual | 2                  |
| Quarterly   | 4                  |
| Monthly     | 12                 |
| Daily       | 365                |
| Hourly      | 365 * 24           |
| Minutely    | 365 * 24 * 60      |
| Second      | 365 * 24 * 60 * 60 |

As we go from annual compounding towards more frequent compounding frequency, we are moving from **discrete compounding** to **continuous compounding**

## Compounding

$$
\begin{align}
\text{FV} &= \text{PV} (1+r)^t \\
&= \text{PV} \left(1 + \frac{r}{m} \right)^{mt} & \text{(Discrete)} \\
&= \text{PV} \times e^{rt} & \text{(Continuous)}
\end{align}
$$

