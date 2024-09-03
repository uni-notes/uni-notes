# Fund Management Evaluation

## Metrics

| Metric                             | Formula                                                      | Meaning                                                      |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Net Selectivity Measure<br />$R_r$ | $R_P - (R_f+R_S+R_U)$                                        | Excess return obtained solely through portfolio optimization |
| Expense Ratio                      | $\dfrac{\text{Total Expenses}}{\text{Net Assets}}$           |                                                              |
| Portfolio Turnover Ratio           | $\dfrac{\min(\text{Purchases}, \text{Sales})}{\text{Assets}}$ | How quickly securities in fund are bought/sold by fund manager |
| Tracking Error                     | $\sigma(R_P-R_B)$                                            | How well portfolio tracks the benchmark                      |

## Fama’s Decomposition of Total Return

$$
R_P = R_f + R_s + R_u + R_r
$$

|       |                                               |                                                              |
| ----- | --------------------------------------------- | ------------------------------------------------------------ |
| $R_s$ | Return from systematic risk                   | $(R_m-R_f) \beta_P$                                          |
| $R_u$ | Return from unsystematic risk                 | $(R_m-R_f)  \left(\dfrac{\sigma_P}{\sigma_m} - \beta_P \right)$ |
| $R_r$ | Residual return/<br />Net Selectivity Measure |                                                              |

## Expense Ratio

| Type    | Typical $\%$   |
| ------- | -------------- |
| Active  | $[0.50, 0.75]$ |
| Passive | $[0.02, 0.20]$ |

Fund with a smaller amount of assets has high expense ratio due to limited funds for covering costs

International funds may have high operational expenses due to staffing in multiple countries

## Portfolio Turnover Ratio

Funds with high PTR will tend to have higher fees to reflect turnover costs

However, high PTR tends to translate higher overall returns, thus mitigating the impact of additional fees

## Goals for Fund Manager

- Minimize $\beta$
- Maximize $\alpha$
- Minimize $\text{ER}$
- Maximize $R_r$