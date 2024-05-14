# Cost of Capital

Depends primarily on the use of funds, not the source, because every investment has a different risk associated with it.

Debt is almost always the cheapest source of capital, but has some trouble associated with it. This will be covered in a future topic.

It is always calculated as WACC (Weighted average cost of capital)

Lower the WACC the better

| Alias Name                  | Perspective of |
| --------------------------- | -------------- |
| Required return             | Investor       |
| Appropriate discount rate   | Firm           |
| Compound rate               | Calculations   |
| Opportunity cost of capital | idk            |

- Cost of equity
- Cost of debt/distress

## Uses

1. WACC is used to value the entire firm
2. Evaluate return for projects
3. Evaluate performance of firm

## Some Notes

- Growing companies have high WACC, as they have risks associated with them
- It is better if WACC decreases over time

## Calculation

$$
\begin{aligned}
\text{WACC} = \quad & w_l  \times k_l (1-\tau) \\
+ & w_b  \times k_b (1-\tau) \\
+ & w_p \times k_p \\
+ & w_c \times k_c
\end{aligned}
$$

|      Term      | Meaning                                  |            Formula            |
| :------------: | ---------------------------------------- | :---------------------------: |
|     $w_d$      | Proportion of debt                       | $\frac{n_d}{n_d + n_p + n_c}$ |
|     $w_p$      | Proportion of preference shares          | $\frac{n_p}{n_d + n_p + n_c}$ |
|     $w_c$      | Proportion of common shares              | $\frac{n_c}{n_d + n_p + n_c}$ |
|     $k_l$      | Pre-Tax Cost of Loan (Interest Rate)     |                               |
| $k_l (1-\tau)$ | Post-Tax Cost of Loan                    |                               |
|     $k_b$      | Pre-Tax Cost of Bond (Yield to Maturity) |                               |
| $k_b (1-\tau)$ | Post-Tax Cost of Bond                    |                               |
|     $k_p$      | Cost of preference shares                |       $\frac{D_p}{P_p}$       |
|     $k_c$      | Cost of common shares                    |                               |
|     $\tau$     | Tax rate                                 |           Available           |

Interest is tax-deductable, hence it gives ‘tax shield’

## CAPM

Capital Asset Pricing Model

Describes relation between systematic risk and expected rate of return of risky investments.

Expected return on a risk investment depends on

- Risk-free rate (return rate of bond)
- Risk premium, depending on $\beta$, where $\beta$ is the sensitivity of the stock wrt the market

$$
\begin{aligned}
k
&= r_\text{min} \\
&= r_f + \beta \Big[ R_m - r_f \Big]
\end{aligned}
$$

where

- $r_\text{min} =$ Required return of investment
- $r_f =$ Risk-Free rate
- $r_m =$ Stock market return
  - Take only recent data (say, 1 year or so)

