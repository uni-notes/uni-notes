# Selection

## Models

|        |                                             |
| ------ | ------------------------------------------- |
| Null   | $\beta_0 + u$                               |
| Subset | $\beta_0 + \sum_j^k' \beta_j x_j + u; k'<k$ |
| Full   | $\beta_0 + \sum_j^k \beta_j x_j + u$        |

## Model Selection

1. Fit multiple models $g_i$ on the training data
2. Use interval validation data for hyper parameter tuning of each model $g_i$
3. Use external validation data for model selection and obtain $g^*$
4. Combine the training and validation data. Refit $g^*$ on this set to obtain $g^{**}$
5. Assess the performance of $g^{**}$ on the test data

Finally, train $g^{**}$ on the entire data to obtain $\hat f$

## Feature Selection

Also called as Subset/Variable Selection

For $p$ potential predictors, $\exists \ 2^p$ possible models. Comparing all subsets is computationally-infeasible

|                             | Selection Type | Constraint Type |                                                              | Advantage                                                    | Disadvantage                                 |
| --------------------------- | -------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| Full Search                 | Discrete       | Hard            | Brute-Force                                                  | Global optima                                                | Computationally-expensive                    |
| Forward stepwise selection  | Discrete       | Hard            | Starts with the null model, and then adds predictors one-at-a-time | Computationally efficient<br />Lower sample size requirement |                                              |
| Backward Stepwise Selection | Discrete       | Hard            | Start with the full model and remove predictors one-at-a-time |                                                              | Expensive<br />Large sample size requirement |
| LASSO                       | Continuous     | Soft            | Refer to regularization                                      |                                                              |                                              |

Neither selection method is guaranteed to find the best subset of predictors.

### Forward

1. Let M0 denote the null model.
2. Fit all univariate models. Choose the one with the best in-sample fit (smallest RSS, highest R2) and add that variable – say x(1)– to M0. Call the resulting model M1.
3. Fit all bivariate models that include x(1): y ∼ β0 + β(1)x(1) + βj xj , and add xj from the one with the best in-sample fit to M1. Call the resulting model M2.
4. Continue until your model selection rule (cross-validation error, AIC, BIC) is lower for the current model than for any of the models that add one variable.

### IDK

- $\alpha_\text{add}$ usually $0.05$ or $0.10$
- $\alpha_\text{remove} > \alpha_\text{add}$

### Forward Stepwise Regression

1. Write down full possible model with all predictors, functions, interactions, etc
2. Regress $y$ against each model term individually
3. Pick $\alpha_\text{add}$ and $\alpha_\text{remove}$ such that $\alpha_\text{add}<\alpha_\text{remove}$
4. Pick best regressor
      1. Calculate $t=\beta_j/\text{SE}(\beta_j)$
      2. Calculate $p$ value for each term
      3. Pick smallest $p$-value $p^*_j$
      4. $p^*_j<\alpha_\text{add} \implies$ add parameter $j$
5. Check if previous term should be removed
   1. For all previously-added regressors, find the one with the lowest $t$ score and hence highest $p$ value $p^*_{j'}$
   2. If $p^*_{j'} > \alpha_\text{remove}$, remove $j'$
6. Repeat step 4-5 until no improvement

### Omitted Variable Bias

If a correct regressor $x_j$ is missing from the model, then the remaining model parameters will be biased if $x_j$ is related to the other vars

The bias will be proportional to the correlation between the missing $x_j$ and the regressors used in the model

## Uncounted DOF

Every time you test a regressor term for the model, it is an addition to the degree of freedom, whether or not you include it in the model

Causes data snooping

DOF = $n-p +$ no of trials