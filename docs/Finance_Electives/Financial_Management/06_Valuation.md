# Valuation

## Debt

It is the rate of return the firm’s lenders demand when they loan money to the firm. 

### Forms of Borrowing

| Type    |                |
| ------- | -------------- |
| Private | Bank Loan      |
| Public  | Bond/Debenture |

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
\text{Bond Price} = \sum_{t=1}^T \frac{\text{Coupon } t}{(1+\text{YTM})^t} + \frac{\text{PAR}}{(1+\text{YTM})^T}
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

- Hybrid of debt and common shares
- Fixed dividends
- Deferrable dividends
- They don’t have voting rights
- There is no expiration date
- It is the only real example of perpetuity
- Usually higher return than bonds

$$
k_p = \frac{d_p}{p_p} \quad \left(\frac{c}{r} \text{ from Perpetuity} \right)
$$

## Common Shares

Returns

- Dividends
- Capital Gains

Difficult to estimate pricing, as there are so many variables in play

1. Unsure cashflows
2. Life of investment is infinite
3. No way to calculate required rate of return

It is frowned upon for a corporation to reduce dividends. Hence, if it increases dividends, it does so very carefully.

### Book value Method

Most appropriate for established companies

### Dividend Growth Model

$$
\begin{aligned}
g &= \text{ROE} \times \text{Retention Rate} \\
D_t &= D_{t-k} \times (1+g)^k \\
\implies
P_t &= \frac{D_{t+1}}{k_\text{CS} - g} \quad \cancel{+ \frac{P_\infty}{(1+r)^\infty}}
\end{aligned}
$$

where

- $g =$ dividend growth rate
  - non-zero constant percentage change of dividend from one year to next. If non-constant, we take average $g$ over a few years
  - $g = \text{ROE} \times b$
    - $\text{ROE \%}=$ Return on Equity
    - $b \% =$ Retention rate, Plowback rate
      - $b \% = 1-\text{Payout Rate}$
- $k=$ market discount rate

| Dividend Growth |   $g$   |            |
| --------------- | :-----: | ---------- |
| No              |   $0$   | Perpetuity |
| Constant        | $\ge 0$ |            |

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
\begin{aligned}
k
&= r_\text{min} \\
&= r_f + \beta \Big( E(r_m) - r_f \Big)
\end{aligned}
$$

where

- $r_\text{min} =$ Required return of investment
- $r_f =$ Risk-Free rate
- $r_m =$ Stock market return
  - Take only recent data (say, 1 year or so)

