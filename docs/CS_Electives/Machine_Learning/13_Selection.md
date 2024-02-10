# Selection

Also called as Subset/Variable Selection

For $p$ potential predictors, $\exists \ 2^p$ possible models. Comparing all subsets is computationally-infeasible

## Models

|      |                                      |
| ---- | ------------------------------------ |
| Null | $\beta_0 + u$                        |
| Full | $\beta_0 + \sum_i^p \beta_i x_i + u$ |

## Selection Methods

|                             | Selection Type | Constraint Type |                                                              | Advantage                                                    | Disadvantage                                 |
| --------------------------- | -------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| Forward stepwise selection  | Discrete       | Hard            | Starts with the null model, and then adds predictors one-at-a-time | Computationally efficient<br />Lower sample size requirement |                                              |
| Backward Stepwise Selection | Discrete       | Hard            | Start with the full model and remove predictors one-at-a-time |                                                              | Expensive<br />Large sample size requirement |
| LASSO                       | Continuous     | Soft            | Refer to regularization                                      |                                                              |                                              |

Neither selection method is guaranteed to find the best subset of predictors.

### Forward

1. Let M0 denote the null model.
2. Fit all univariate models. Choose the one with the best in-sample fit (smallest RSS, highest R2) and add that variable – say x(1)– to M0. Call the resulting model M1.
3. Fit all bivariate models that include x(1): y ∼ β0 + β(1)x(1) + βj xj , and add xj from the one with the best in-sample fit to M1. Call the resulting model M2.
4. Continue until your model selection rule (cross-validation error, AIC, BIC) is lower for the current model than for any of the models that add one variable.