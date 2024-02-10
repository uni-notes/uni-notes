# Tuning

## Generalization

The ability of trained model to be able to perform well on unseen inputs. Better validation result $\implies$ Better generalization

Note: Always try to overfit with a very small sample and then focus on generalization

### Generalization & Training Size

Generalization improves with size of training set, until a saturation point, after which it stops improving.

|                | More data $\implies$                                        |
| -------------- | ----------------------------------------------------------- |
| Parametric     | asymptote to an error value exceeding Bayes error           |
| Non-Parametric | better generalization until best possible error is achieved |

![image-20230401142609958](./assets/image-20230401142609958.png)

### Bias-Variance Tradeoff

Usually U-Shaped

![image-20230401141618389](./assets/image-20230401141618389.png)

## Regularization

Techniques to address overfitting

### Regularization Penalty

Reduce errors by fitting the function appropriately on the given training set, to help reduce variance, and hence avoid overfitting, while minimally affecting bias.

Not that this is only possible if all the input variables are standardized.

This is done by adding a penalty term in the cost function.

$$
J'(\theta) = J(\theta) + \dfrac{\textcolor{hotpink}{\text{Regularization Penalty}}}{\text{Sample Size}}
$$

|                         | Effect                                      | Goal                                  | Penalty|
|---                      | ---                                         | ---                                   | ---|
|$L_1$<br />(Lasso)       | Encourages sparsity<br />Eliminates feature(s) completely | Feature selection                     | $\lambda \sum \limits_{j=0}^k \dfrac{\vert\beta_j - \beta^*_j \vert}{\sigma^2_{\beta^*_j}}$|
|$L_2$<br />(Rigde)       | Reduce effect of large coefficients | Scale down the coefficients           | $\lambda \sum \limits_{j=0}^k \dfrac{(\beta_j - \beta^*_j)^2}{\sigma^2_{\beta_j^*}}$|
|$L_3$<br />(Elastic Net) |                                             |                                       | $\alpha L_1 + (1-\alpha) L_2$|
|Entropy                  | Encourages sparsity<br />Cause high variation in between parameters | Encourage parameters to be different<br /> | $\lambda \sum \limits_{j=0}^k - P(\beta_j) \ln P(\beta_j)$ |

where

- $\beta^*_j$ is the prior-known most probable value of $\beta_j$
- $\sigma^2_{\beta^*_j}$ is the prior-known standard deviation of $\beta_j$

These 2 incorporate desirable Bayesian aspects in our model.

Example

|    $y$    | $\hat \beta$ |
| :-------: | :----------: |
| $\beta x$ |      0       |
| $x^\beta$ |      1       |

Incorporating $\hat \beta$ into the regularization incorporates maximum likelihood estimate of the coefficients.

Contours of where the penalty is equal to 1 for the three penalties L1, L2 and elastic-net

![image](./assets/a3bde8dd-8d3d-4b34-b5b8-cfee29c7c464.png)

### Increase DOF

- Reduce $k$

  - Feature Selection

  - Dimensionality Reduction

- Increase $n$

### Subsampling at each iteration

- columns
- rows

### IDK

![image-20240106160827581](./assets/image-20240106160827581.png)

### Early-Stopping

![Training vs. Validation Error for Overfitting](./assets/error_graph.png)

### Dropout



### Ensembling

## Hyper-Parameter Tuning

### Manual



### Evolutionary Computing



### Bayesian

|                |                                                              | Advantage | Disadvantage   |
| -------------- | ------------------------------------------------------------ | --------- | -------------- |
| Manual         |                                                              |           | Time-Consuming |
| Grid Search    |                                                              |           |                |
| Random Search  |                                                              |           |                |
| Evolutionary   | Randomization, Natural Selection, Mutation                   |           |                |
| Bayesian       | Probabilistic model of relationship b/w cost function and hyper-parameters, using information gathered from trials |           |                |
| Gradient-Based | Treat hyper parameter tuning like parameter fitting          |           |                |
| Early-Stopping | Focus resources on settings that look promising<br />eg: Successive Halving |           |                |

