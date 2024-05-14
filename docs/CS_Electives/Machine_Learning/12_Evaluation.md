# Model Evaluation

## Goals

1. Minimize bias
2. Minimize variance
3. Minimize generalization gap

## Guidelines

- Always check if your model is able to learn from a synthetic dataset where you know the underlying data-generating process
- Metrics computed from test set may not be representative of the true population
- Always look at multiple metrics; never trust a single one alone
  - false positives and false negatives are seldom equivalent
  - understand the problem to known the right tradeoff
- Always monitor the worst-case prediction
  - Maximum loss
  - 95th percentile loss


## Baseline/Benchark models

Always establish a baseline

- Basic/Naive/Dummy predictions
  - Regression
    - Mean
    - Max
    - Min
    - Random
  - Classification
    - Mode
    - Random
  - Time series specific
    - Persistence
      - $\hat y_{t+h} = y_t$
      - Latest value available
      - Great for short horizons
    - Climatology
      - $\hat y_{t+h} = \bar y_{i \le t}$
      - Average of all observations until present
      - Great for short horizons
    - Combination of Persistence and Climatology
      - $\hat y_{t+h} = \beta_1 y_t + \beta_2 \bar y_{i \le t}$
    - Lag: $\hat y_{t+h} = y_{t-k}$
    - Seasonal Lag: $\hat y_{t+h} = y_{t+h-m}$
    - Moving average
    - Exponential average
- Human Level Performance
- Literature Review
- Performance of older system

### Significance

All the evaluation should be performed relative to the baseline.

For eg: Relative RMSE = RMSE(model)/RMSE(baseline), with “good” threshold as 1

## Probabilistic Evaluation

Now, we need to see if any difference in accuracy across models/hyperparameters is statistically-significant, or just a matter of chance.

Summary Statistics: Don’t just look at the mean evaluation metric of the multiple splits; also get the uncertainty associated with the validation process.

- Standard error of accuracy estimate
- Standard deviation
- Quantiles
- PDF
- VaR

## Bessel’s Correction

$$
\begin{aligned}
\text{Metric}_\text{corrected} &= \text{Metric}_\text{uncorrected} \times \dfrac{n}{\text{DOF}} \\
\text{DOF} &= n-k-e
\end{aligned}
$$

- where
  - $n=$ no of samples
  - $k=$ no of parameters
  - $e=$ no of intermediate estimates (such as $\bar x$ for variance)
- Do not perform this on metrics which are already corrected for degree of freedom (such as $R^2_\text{adj}$)
- Modify accordingly for squares/root metrics

## Regression Evaluation

| Metric                                                      |                           Formula                            |   Unit   |     Range     | Preferred Value | Signifies                                                    | Advantages<br />✅                                            | Disadvantages<br />❌                                         | Comment                                                      |
| :---------------------------------------------------------- | :----------------------------------------------------------: | :------: | :-----------: | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------- |
| $R^2$<br />(Coefficient of Determination)                   |                       $1 - \text{RSE}$                       | Unitless |   $[0, 1]$    | $1$             | Proportion of changes in dependent var explained by regressors.<br /><br />Proportion of variance in $y$ explained by model wrt variance explained by mean<br /><br/>Demonstrates ___ of regressors<br/>- Relevance<br/>- Power<br/>- Importance |                                                              | Cannot use to compare same model on different samples, as it depends on variance of sample<br /><br />Susceptible to spurious regression, as it increases automatically when increasing predictors |                                                              |
| Adjusted $R^2$                                              |      $1 - \left[\dfrac{(n-1)}{(n-k-1)} (1-R^2)\right]$       | Unitless |   $[0, 1]$    | $1$             |                                                              | Penalizes large number of predictors                         |                                                              |                                                              |
| Accuracy                                                    |                     $100 - \text{MAPE}$                      |    %     |  $[0, 100]$   | $100 \%$        |                                                              |                                                              |                                                              |                                                              |
| $\chi^2_\text{reduced}$                                     | $\dfrac{\chi^2}{n-k} = \dfrac{1}{n-k}\sum \left( u_i/\sigma_i \right)^2$ |          | $[0, \infty]$ | $1$             |                                                              |                                                              |                                                              | $\approx 1:$ Good fit<br />$\gg 1:$ Underfit/Low variance estimate<br />$\ll 1:$ Overfit/High variance estimate |
| Spearman’s Correlation                                      | $\dfrac{ r(\ rg( \hat y), rg(y) \ ) }{ \sigma(\ rg(\hat y) \ ) \cdot \sigma( \ rg(y) \ ) }$ | Unitless |   $[-1, 1]$   | $1$             |                                                              | Very robust against outliers<br />Invariant under monotone transformations of $y$ |                                                              |                                                              |
| DW<br />(Durbin-Watson Stat)                                |                                                              |          |               | $> 2$           | Confidence of error term being random process                |                                                              | Not appropriate when $k>n$                                   | Similar to $t$ or $z$ value<br />If $R^2 >$ DW Statistic, there is [Spurious Regression](#Spurious Regression) |
| AIC<br />Akaike Information Criterion                       |                       $-2 \ln L + 2k$                        |          |               | $0$             | Leave-one-out cross validation score                         | Penalizes predictors more heavily than $R_\text{adj}^2$      | For small values of $n$, selects too many predictors<br /><br />Not appropriate when $k>n$ |                                                              |
| AIC Corrected                                               |            $\text{AIC} + \dfrac{2k(k+1)}{n-k-1}$             |          |               | $0$             |                                                              | Encourages feature selection                                 |                                                              |                                                              |
| BIC/SBIC/SC<br />(Schwarz’s Bayesian Information Criterion) |                     $-2 \ln L + k \ln n$                     |          |               | $0$             |                                                              | Penalizes predictors more heavily than AIC                   |                                                              |                                                              |
| HQIC<br />Hannan-Quinn Information Criterion                |            $-2 \ln L + 2k \ln \vert \ln n \vert$             |          |               | $0$             |                                                              |                                                              |                                                              |                                                              |

### Probabilistic Evaluation

You can model the error such as MAE as a $\chi^2$ distribution with dof = $n-k$

The uncertainty can be obtained from the distribution

### Spurious Regression

Misleading statistical evidence of a relationship that does not truly exist

Occurs when we perform regression between

- 2 independent variables

  and/or

- 2 non-stationary variables

  (Refer econometrics)

You may get high $R^2$ and $t$ values, but $u_t$ is not white noise (it is non-stationary)

$\sigma^2_u$ becomes infinite as we go further in time

## Classification Evaluation

There is always a tradeoff b/w specificity and sensitivity

| Metric                                                       |                           Formula                            | Preferred Value |     Unit     |                            Range                             | Meaning                                                      |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :-------------: | :----------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Entropy of each classification                               |               $H_i = -\sum \hat y \ln \hat y$                |  $\downarrow$   |              |                        $[0, \infty)$                         | Uncertainty in a single classification                       |
| Mean Entropy                                                 |               $H_i = -\sum \hat y \ln \hat y$                |                 |              |                                                              | Uncertainty in classification of entire dataset              |
| **Accuracy**                                                 | $1 - \text{Error}$<br />$\dfrac{\text{TP + TN}}{\text{TP + FP + TN + FN}}$ |   $\uparrow$    |      %       |                          $[0, 100]$                          | $\dfrac{\text{Correct Predictions}}{\text{No of predictions}}$ |
| **Error**                                                    |      $\dfrac{\text{FP + FN}}{\text{TP + FP + TN + FN}}$      |    $[0, 1]$     | $\downarrow$ | $\dfrac{\text{Wrong Predictions}}{\text{No of predictions}}$ |                                                              |
| **F Score**<br />F~1~ Score<br />F-Measure                   | $\dfrac{2}{\dfrac{1}{\text{Precision}} + \dfrac{1}{\text{Recall}}}$<br />$2 \times \dfrac{P \times R}{P + R}$ |   $\uparrow$    |   Unitless   |                           $[0, 1]$                           | Harmonic mean between precision and recall<br />Close to lower value |
| ROC-AUC<br />Receiver-Operator Characteristics-Area Under Curve |     Sensitivity vs (1-Specificity)<br />= (1-FNR) vs FPR     |   $\uparrow$    |   Unitless   |                           $[0, 1]$                           | AUC = Probability that algo ranks a +ve over a -ve<br />Robust to unbalanced dataset |
| **Precision**<br />PPV/<br />Positive Predictive Value       | $\dfrac{\textcolor{hotpink}{\text{TP}}}{\textcolor{hotpink}{\text{TP}} + \text{FP}}$ |   $\uparrow$    |   Unitless   |                           $[0, 1]$                           | How many actual +ve values were correctly predicted as +ve   |
| **Recall**<br />Sensitivity<br />True Positive Rate          | $\dfrac{\textcolor{hotpink}{\text{TP}}}{\textcolor{hotpink}{\text{TP}} + \text{FN}}$ |   $\uparrow$    |   Unitless   |                           $[0, 1]$                           | Out of actual +ve values, how many were correctly predicted as +ve |
| **Specificity**<br />True Negative Rate                      | $\dfrac{\textcolor{hotpink}{\text{TN}}}{\textcolor{hotpink}{\text{TN}} + \text{FP}}$ |   $\uparrow$    |   Unitless   |                           $[0, 1]$                           | Out of actual -ve values, how many were correctly predicted as -ve |
| NPV<br />Negative Predictive Value                           | $\dfrac{\textcolor{hotpink}{\text{TN}}}{\textcolor{hotpink}{\text{TN}} + \text{FN}}$ |                 |   Unitless   |                           $[0, 1]$                           | Out of actual -ve values, how many were correctly predicted as -ve |
| $F_\beta$ Score                                              | $\dfrac{(1 + \beta^2)}{{\beta^2}} \times \dfrac{P \times R}{P + R}$ |   $\uparrow$    |   Unitless   |                            [0, 1]                            | Balance between importance of precision/recall               |
| **FP Rate**                                                  | $\begin{aligned}\alpha &= \dfrac{\textcolor{hotpink}{\text{FP}}}{\textcolor{hotpink}{\text{FP}} + \text{TN}} \\ &= 1 - \text{Specificity} \end{aligned}$ |  $\downarrow$   |   Unitless   |                           $[0, 1]$                           | Out of the actual -ve, how many were misclassified as Positive |
| **FN Rate**                                                  | $\begin{aligned}\beta &= \dfrac{\textcolor{hotpink}{\text{FN}}}{\textcolor{hotpink}{\text{FN}} + \text{TP}} \\ &= 1 - \text{Sensitivity} \end{aligned}$ |  $\downarrow$   |   Unitless   |                           $[0, 1]$                           | Out of the actual +ve, how many were misclassified as Negative |
| Balanced Accuracy                                            |         $\frac{\text{Sensitivity + Specificity}}2{}$         |                 |   Unitless   |                                                              |                                                              |
| MCC<br />Mathews Correlation Coefficient                     | $\dfrac{\text{TP} \cdot \text{TN} - \text{FP}\cdot \text{FN} }{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$ |   $\uparrow$    |   Unitless   |                          $[-1, 1]$                           | 1 = perfect classification<br />0 = random classification<br />-1 = perfect misclassification |
| Markdedness                                                  |                        PPV + NPV - 1                         |                 |              |                                                              |                                                              |
| Brier Score Scaled                                           |                                                              |                 |              |                                                              |                                                              |
| Nagelkerke’s $R^2$                                           |                                                              |                 |              |                                                              |                                                              |
| Hosmer-Lemeshow Test                                         |                                                              |                 |              |                                                              | Calibration: agreement b/w obs and predicted                 |

### Graphs

| Graph                   |                                                              |                                                              | Preferred                                     |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| Error Rate              | ![image-20240220125218210](./assets/image-20240220125218210.png) |                                                              |                                               |
| ROC Curve               | ![image-20240704131040197](./assets/image-20240704131040197.png) | How does the classifier compare to a classifier that predicts randomly with $p=\text{TPR}$<br />How well model can discriminate between $y=0$ and $y=1$ | Top-Left<br />At least higher than 45deg line |
| Calibration Graph       | ![image-20240528142438139](./assets/image-20240528142438139.png) | Create bins of different predicted probabilities<br />Calculate the fraction of $y=1$ for each bin<br />Confidence intervals (more uncertainty for bins with fewer samples)<br />Histogram showing fraction of samples in each bin | Along 45deg line                              |
| Confusion Probabilities | ![image-20240718114749534](./assets/image-20240718114749534.png) |                                                              |                                               |

### Tradeoff for Threshold

![image-20240704130932826](./assets/image-20240704130932826.png)

![image-20240704130941679](./assets/image-20240704130941679.png)

### Probabilistic Evaluation

Wilson score interval

![img](./assets/Wilson_score_interval_and_logistic_example.png)

You can model accuracy as a binomial distribution with

- $n=$ Validation set size
  - = No of predictions
  - = No of k folds * Validation Set Size
- $p=$ Obtained Accuracy of classifier

Similar to confidence intervals for proportion

The uncertainty can be obtained from the distribution

![image-20240106202910165](./assets/image-20240106202910165.png)

```python
for n in [100, 1_000, 10_000, 100_000]:
  dist = stats.binom(n, 0.7)
  
  alpha = 0.025
  
  interval_width = dist.isf(alpha) - dist.isf(1-0.975)
  print(f"Size: {interval_width/n * 100}")
  # returns alpha % of observed accuracy that fall outside the inverval --> This is the maximum further accuracy that is theoretically achievable
```

### Decision Boundary

Plot random distribution of values

For eg:

![0](./assets/9bca0d386fe78d1cbd051112ed2f8f1f69a70ee95971fae928749471.png)

### Confusion Matrix

$n \times n$ matrix, where $n$ is the number of classes

#### Binary Classification

![confusion_matrix_True_False_Positive_Negative](./assets/confusion_matrix.png){ loading=lazy }

#### Multi-Class Classification

Confusion Matrix with respect to A

|      | A    | B    | C    |
| ---- | ---- | ---- | ---- |
| A    | TP   | FN   | FN   |
| B    | FP   | TN   | TN   |
| C    | FP   | TN   | TN   |

### Classification Accuracy Measures

#### Jacquard Index

$$
\begin{aligned}
J(y, \hat y)
&= \frac{|y \cap \hat y|}{|y \cup \hat y|} \\
&= \frac{|y \cap \hat y|}{|y| + |\hat y| - |y \cap \hat y|}
\end{aligned}
$$

### Multi-Class Averaging

|                  |                                                             |                                                     |
| ---------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| Micro-Average    | All samples equally contribute to average                   | $\dfrac{1}{C}\sum_{c=1}^C \dfrac{\dots}{\dots}$     |
| Macro-Average    | All classes equally contribute to average                   | $\dfrac{\sum_{c=1}^C \dots}{\sum_{c=1}^C \dots}$    |
| Weighted-Average | Each class’ contribution to average is weighted by its size | $\sum_{c=1}^C \dfrac{n_c}{n}  \dfrac{\dots}{\dots}$ |

## Inspection

| Inspection             | Identify          |
| ---------------------- | ----------------- |
| Error Analysis         | Systematic errors |
| Bias-Variance Analysis | General errors    |

## Error Analysis

Residual Inspection

Perform all the inspection on

- train and dev data
- externally-studentized residuals, to correct for leverage

There should be no explainable unsystematic component

- Symmetric distribution for values of error terms **for a given value $x$**
- **Not** over time/different values of $x$
- This means that
  1. you have used up all the possible factors
  2. $u_i$ only contains the non-systematic component

| Ensure                                                       | Meaning                                                      | Numerical                                                    | Graphical                                                    | Implication if violated                                      | Action if violated                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Random residuals                                             | - No relationship between error and independent variables<br />- No relationship between error and predictions<br/>- If there is correlation, $\exists$ unexplained system component | Normality test<br />$E(a |b) = 0$<br />$r(a, b) = 0$<br/>$a \in [u_i, \vert u_i \vert , u_i^2]$<br/>$b \in [ x_i, \vert x_i \vert , x_i^2, y_i, \vert y_i \vert , y_i^2 ]$ | Q-Q Plot<br />Histogram<br /><br />$u_i-x_i$<br />$\vert u_i \vert -x_i$<br />$u_i^2-x_i$<br /><br />$u_i-y_i$<br />$\vert u_i \vert -y _i$<br />$u_i^2-y_i$ | ❌ Unbiased parameters                                        | Fix model misspecification                                   |
| No autocorrelation                                           | - Random sequence of residuals should bounce between +ve and -ve according to a binomial distribution<br />- Too many/few bounces may mean that sequence is not random<br /><br />No [autocorrelation](#Autocorrelation) between $u_i$ and $u_j$ | $r(u_i, u_j \vert x_i, x_j )=0$<br />Runs test<br />DW Test  | Sequence Plot of $u_i$ vs $t$<br />Lag Plot of $u_i$ vs $u_j$ | ✅ Parameters remain unbiased<br />❌ MSE estimate will be lower than true residual variance<br /><br />Properties of error terms in presence of $AR(1)$ autocorrelation <br/><br/>- $E[u_i]=0$<br/>- $\text{var}(u_i)= \sigma^2/(1-\rho^2)$<br/>- $r(u_i, u_{i-p}) = \rho^p \text{var}(u_i) = \rho^p \sigma^2/(1-\rho^2)$ | Fix model misspecification<br />Incorporate trend<br />Incorporate lag (last resort)<br />Autocorrelation analysis |
| No effect of outliers                                        |                                                              |                                                              |                                                              |                                                              | Outlier removal/adjustment                                   |
| Low leverage & influence of each point                       |                                                              |                                                              |                                                              |                                                              | Data transformation                                          |
| Homoscedasticity<br />(Constant variance)                    | $\sigma^2 (u_i|x_i) = \text{constant}$ should be same $\forall i$ |                                                              |                                                              |                                                              | Weighted regression                                          |
| Error in input variables                                     |                                                              |                                                              |                                                              |                                                              | Total regression                                             |
| Correct model specification                                  |                                                              |                                                              |                                                              |                                                              | Model building                                               |
| Goodness of fit                                              |                                                              | - MLE Percentiles<br/>- Kolmogorov Smirnov                   |                                                              |                                                              |                                                              |
| Significance in difference in residuals for models/baselines | Ensure that all the models are truly different, or is the conclusion that one model is performing better than another due to chance | [Comparing Samples](../../1_Core/Probability_&_Statistics/10_Comparing_Samples.md) |                                                              |                                                              |                                                              |

### Why is this important?

For eg: Running OLS on Anscombe’s quartet gives

- same curve fit for all
- Same $R^2$, RMSE, standard errors for coefficients for all

But clearly the fit is not equally optimal

1. Only the first one is acceptable
2. Model misspecification
3. Outlier
4. Leverage

![image-20240217123508539](./assets/image-20240217123508539.png)

which is shown in the residual plot

![image-20240217123916483](./assets/image-20240217123916483.png)

### Aggregated Inspection

- Aggregate the data based on metadata
- Evaluate metrics on groups

$$
u_i \vert g(x_i) \\
u_i \vert g(y_i)
$$

where $g()$ is the group, which can be $x_{ij}, y_i$ or combination of these

- Image blurry/flipped/mislabelled
- Gender
- Age
- Age & Gender

### Diebold-Mariano Test

Determine whether predictions of 2 models are significantly different

Basically the same as [Comparing Samples](../../1_Core/Probability_&_Statistics/10_Comparing_Samples.md) for residuals

## Bias-Variance Analysis

### Evaluation Curves

Related to [Interpretation](#Interpretation)

- Always look at all curves with **uncertainties** wrt each epoch, train, hyper-parameter value.
- The uncertainty in-sample and out-sample should also be similar
  - If train set metric uncertainty is small and test set metric uncertainty is large, this is bad even if the average loss metric is same

|                      | Learning Curve                                               | Loss Curve                                                 | Validation Curve                              |
| -------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------- |
| Loss vs              | Train Size                                                   | Epochs                                                     | Hyper-parameter value                         |
| Comment              | Train Error increases with train size, because model overfits small train datasets |                                                            |                                               |
| Purpose: Detect      | Bias<br />Variance<br />Utility of adding more data          | Optimization problems<br />Undertraining<br />Overtraining | Model Complexity<br />Optimal Hyperparameters |
| No retraining        | ❌                                                            | ✅                                                          | ❌                                             |
| No extra computation | ❌                                                            | ✅                                                          | ❌                                             |

### Learning Curve

Based on the slope of the curves, determine if adding more data will help

|                                                              | Conclusion                       |
| ------------------------------------------------------------ | -------------------------------- |
| ![image-20240409104640019](./assets/image-20240409104640019.png) | High Bias<br />(Underfitting)    |
| ![image-20240409104925429](./assets/image-20240409104925429.png) | High Variance<br />(Overfitting) |
| ![image-20240409105039243](./assets/image-20240409105039243.png) | High Bias<br />High Variance     |

### Loss Curve

![image-20240214082637449](./assets/generalization_complexity.svg)

![image-20240409105521534](./assets/image-20240409105521534.png)

#### Same Model, Variable Learning Rate

![image-20240409105708117](./assets/image-20240409105708117.png)

### Validation Curve

![image-20240716115219646](./assets/image-20240716115219646.png)

## Neural Network

|                                         | Recommended Value |      |
| --------------------------------------- | ----------------- | ---- |
| Activation distributions                | $N(0, 1)$         |      |
| Activation gradient distributions       | $N(0, 1)$         |      |
| Parameter weight distributions          | $N(0, 1)$         |      |
| Parameter gradient distributions        | $N(0, 1)$         |      |
| Parameter gradient:weight distributions | $1$               |      |
| Parameter update:weight distributions   | $10^{-3}$         |      |

