# Model Interpretation

Association $\ne$ Causation

## Classification of Inference Techniques

- IDK
  - Model-Specific
  - Model-Agnostic
- Scope
  - Global: Explanation for entire dataset
  - Local: Explanation for single data point

## Inference Techniques

|                                                             | IDK            | Scope  |                                                              |
| ----------------------------------------------------------- | -------------- | ------ | ------------------------------------------------------------ |
| Simple Linear Regression<br />$y = \beta_0 + \beta_j x_j$   | Model-Specific | Global | For every unit increase in $x_j$, $y$ changes by $\beta_j$ units |
| $\ln \vert y \vert = \beta_0 + \beta_j x_j$                 | Model-Specific | Global | For every unit increase in $x_j$, percentage change in $y$ is $\beta_j$ units |
| $\ln \vert y \vert = \beta_0 + \beta_j \ln \vert x_j \vert$ | Model-Specific | Global | Elasticity of $y$ wrt $x_j$ is given by $\beta_j$<br />$\beta_j = \dfrac{\% \Delta y}{\% \Delta x_j}$ |
| SAGE                                                        | Model-Agnostic | Global |                                                              |
| Variable/Feature Importance                                 | Model-Agnostic | Global | Decrease of in-sample error due to splits over $x$, averaged over all trees of ensemble |
| Partial Dependence                                          | Model-Agnostic | Global | Partial derivative of $y$ wrt $x$: Marginal effect of $x$ on $y$ after integrating out all other vars |
| SHAP                                                        | Model-Agnostic | Local  |                                                              |
| LIME                                                        | Model-Agnostic | Local  |                                                              |

