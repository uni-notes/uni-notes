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

## Debt

It is the rate of return the firm’s lenders demand when they loan money to the firm. 

### Forms of Borrowing

| Type    |                |
| ------- | -------------- |
| Private | Bank Loan      |
| Public  | Bond/Debenture |

### Sources of Corporate Debt

- Sorted in order of least risky to more risky
- Also, Sorted in order of least return to highest return

| Source                                          | Duration   | Market |
| ----------------------------------------------- | ---------- | ------ |
| Treasury Bill<br />(T-Bill)                     | Short term | Money  |
| Govt Bond                                       | Long term  |        |
| Corporate Bond (Zero Coupon)<br />(More traded) | Long-term  |        |
| Corporate Bond (Coupon)                         | Long-term  |        |
| Commercial Paper                                | Short-term |        |
| CD (Certificate of Deposit)                     |            |        |
| Rapport                                         |            |        |

### Capital vs Money Market

|                | Duration   | Cost      | Example                                                   |
| -------------- | ---------- | --------- | --------------------------------------------------------- |
| Money Market   | Short-term | Cheaper   | Credit cards<br />Rapport<br />Reverse rapport<br />TBill |
| Capital Market | Long-term  | Expensive |                                                           |

### Bond

Certificate of 

| Term                                  |  Fixed?   | Meaning                                                      | Formula                                                      | Unit            |
| ------------------------------------- | :-------: | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- |
| Face/PAR/Book Value                   |     ✅     | Listing price of the security                                | $\text{PAR } = \frac{\text{Total Amount}}{\text{No of bonds}}$ | Currency        |
| Coupon Rate                           |     ✅     | Interest rate                                                |                                                              | % of face value |
| Time to Maturity/<br />Time to Expiry |     ✅     | Bounding time period by which face value will be repayed<br />(at every payment instant, we only pay the coupon amount) |                                                              |                 |
| Credit Rating                         | Partially |                                                              |                                                              |                 |
| YTM<br />(Yield-to-Maturity)          |           | IRR of the bond<br />Actual return for the buyer of the bond |                                                              |                 |

| Bond Traded at |         Purchase          |      Returns      |
| -------------- | :-----------------------: | :---------------: |
| PAR            | Market Value = Face Value | YTM = Coupon rate |
| Premium        | Market Value > Face Value | YTM < Coupon rate |
| Discount       | Market Value < Face Value | YTM > Coupon rate |

#### Bond Price

$$
\text{Bond Price} = \sum_{t=0}^T \frac{\text{Coupon } t}{(1+\text{YTM})^t} + \frac{\text{PAR}}{(1+\text{YTM})^T}
$$

$$
\text{Bond Price } \propto \frac{1}{\text{Interest Rate}}
$$

This is because, if interest rate increases, lenders will go to loan market, and everyone will sell their bonds.

### Misc

#### Run-on-the-bank

Banks should have minimum liquidity, to ensure that

- If a private bank falls show on SLR, they can request from government, using [Rapport](#Rapport)
- If a govt bank falls show on SLR, they can request from government, using [Reverse Rapport](#Rapport)

#### Rapport

Repurchase agreement

#### Reverse Rapport

### Why are Govt Bonds Risk-Free?

Chance of default is lowest.

## Preference Shares

Hybrid of debt and common shares

Fixed dividends

Deferrable dividends

They don’t have voting rights

There is no expiration date

It is the only real example of perpetuity

Usually higher return than bonds
$$
k_p = \frac{d_p}{p_p} \quad \left(\frac{c}{r} \text{ from Perpetuity} \right)
$$

## Common Shares

### Returns

- Dividents
- Capital Gains

### Pricing

Difficult to estimate pricing, as there are so many variables in play

1. Unsure cashflows
2. Life of investment is infinite
3. No way to calculate required rate of return

It is frowned upon for a corporation to reduce dividends. Hence, if a increases dividends, it does so very carefully.

### Dividend Growth Model

$$
\begin{aligned}
D_t &= D_{t-1} \times (1+g)^t \\
\implies P_t &= \frac{D_{t+1}}{k_\text{CS}} - g \quad \cancel{+ \frac{P_\infty}{(1+r)^\infty}}
\end{aligned}
$$

where $g$ is a non-zero constant percentage change of dividede from one year to next. If non-constant, we take average $g$ over a few years.

|                           |   $g$   |
| ------------------------- | :-----: |
| Zero-Growth Dividends     |   $0$   |
| Constant-Growth Dividends | $\ge 0$ |

#### Limitations

- Assumes constant growth
- Only works when $g \ne 0$

### CAPM

Capital Asset Pricing Model

Describes relation between systematic risk and expected rate of return of risky investments.

Expected return on a risk investment depends on

- Risk-free rate (return rate of bond)
- Risk premium, depending on $\beta$, where $\beta$ is the sensitivity of the stock wrt the market

$$
r_\text{min} = r_f + \beta \Big( E(r_m) - r_f \Big)
$$

where

- $r_\text{min} =$ Required return of investment
- $r_f =$ Risk-Free rate
- $r_m =$ Stock market return
  - Take only recent data (say, 1 year or so)

## Risk

| Types of Risk               | Applicable to all corporations | Risk compensation required? |
| --------------------------- | :----------------------------: | :-------------------------: |
| Systematic                  |               ✅                |              ✅              |
| Unsystematic/<br />Specific |               ❌                |              ❌              |

Portfolio should be created in a way that unsystematic risk is overcome, by picking stocks ideally be negatively-correlated with each other
