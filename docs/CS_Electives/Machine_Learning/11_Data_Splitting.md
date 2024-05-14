# Data Splitting

## Train-Test

The training set has an optimistic
bias, since it is used to choose a hypothesis that looks good on it. Hence, we require a unseen set as it is not biased

Once a data set has been used in the learning/validation process, it is “**contaminated**” – it obtains an optimistic (deceptive) bias, and the error calculated on the data set no longer has the tight generalization bound.

To simulate deployment, any data used for evaluation should be treated as if it does not exist at the time of modelling

## Data Split Sets


|                                                              |                            Train                             |            Development<br />(Inner Validation)            |            Validation<br />(Outer Validation)             |                     Test<br />(Holdout)                      |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: |
| Recommend split %                                            |                              40                              |                            20                             |                            20                             |                              20                              |
| In-Sample<br />(‘Seen’ by model)                             |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| EDA<br />(‘Seen’ by analyst)                                 |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| Pre-Processing “learning”<br />(Normalization, Standardization, …) |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| Feature Engineering “learning”<br />(Selection, Transformation, …) |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| Underfit Evaluation                                          |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| Model Tuning                                                 |                              ✅                               |                             ❌                             |                             ❌                             |                              ❌                               |
| Overfit Evaluation                                           |                              ❌                               |                             ✅                             |                             ❌                             |                              ❌                               |
| Hyperparameter Tuning                                        |                              ❌                               |                             ✅                             |                             ❌                             |                              ❌                               |
| Model Comparison/Selection                                   |                              ❌                               |                             ❌                             |                             ✅                             |                              ❌                               |
| Performance Reporting                                        |                              ❌                               |                             ❌                             |                             ❌                             |                              ✅                               |
| $\hat f$                                                     |                     ${\hat f}_\text{in}$                     |                   ${\hat f}_\text{dev}$                   |                   ${\hat f}_\text{val}$                   |                    ${\hat f}_\text{test}$                    |
| $\hat f$ trained on                                          |                            Train                             |                           Train                           |                         Train+Dev                         |                        Train+Dev+Val                         |
| $E$                                                          |                        $E_\text{in}$                         |                      $E_\text{dev}$                       |                      $E_\text{val}$                       |                       $E_\text{test}$                        |
| Error Names                                                  | Training error/<br />In-Sample Error/<br />Empirical Error/<br />Empirical Risk |                     Development Error                     |                     Validation Error                      | $\hat E_\text{out}$<br />Expected error<br />Prediction error<br />Risk |
| ${\vert H \vert}_\text{set}$                                 |                           $\infty$                           |                       $d_\text{vc}$                       |               ${\vert H \vert}_\text{val}$                |                             $1$                              |
| Comment                                                      |                                                              |                                                           |    Used for “training” on “finalist” set of hypotheses    |       Should not be used for any model decision making       |
| Color Scheme Below                                           |   <span style="background:green;color:white">Green</span>    | <span style="background:yellow;color:black">Yellow</span> | <span style="background:orange;color:white">Orange</span> |     <span style="background:Red;color:white">Red</span>      |

$$
\begin{aligned}
\mathbb{E}[E_\text{test}]
&= E_\text{out} \\
\text{var}[E_\text{test}]
&= \dfrac{\sigma^2_{u}}{n_\text{test}} \\
\end{aligned}
$$

$$
E_\text{out} \le E_\text{set} + O \left( \sqrt{\dfrac{\ln {\vert H \vert}_\text{set}}{n_\text{set}}} \right)
$$

## Test-Size Tradeoff

$$
E_\text{out}(\hat f)
\underbrace{\approx}_\mathclap{n^*_\text{test} \downarrow}
E_\text{out}(\hat f_\text{test})
\underbrace{\approx}_\mathclap{n^*_\text{test} \uparrow}
E_\text{test}(\hat f_\text{test})
$$

|                                                              | Small                                       | Large                                    |
| ------------------------------------------------------------ | ------------------------------------------- | ---------------------------------------- |
| Low Model Bias                                               | ✅                                           | ❌                                        |
| Small Generalization Bound                                   | ❌                                           | ✅                                        |
| Reliable $\hat E_\text{out}$<br />$E_\text{out}(\hat f_\text{test})-E_\text{test}(\hat f_\text{test})$ | ❌                                           | ✅                                        |
| Tested model and final model are same<br />Small $E_\text{out}(\hat f) - E_\text{out}(\hat f_\text{test})$ | ✅                                           | ❌                                        |
| Extreme case<br />Model performance reporting                | “with no certainty, the model is excellent” | “with high certainty, the model is crap” |

![image-20240627171759662](./assets/image-20240627171759662.png)

## Usage

1. Training Data
   1. Get $E_\text{in}$
   2. Overfit all models
   3. Beat baseline model(s)
2. Dev data
   1. Get $E_\text{dev}$
   2. Tune all models to generalize
   3. Beat baseline model(s)
3. Validation data
   1. Compare all models on $E_\text{val}$
   2. Must beat baseline model(s)
   3. Select best model $\hat f_\text{val}^*$
4. Get accuracy estimate of $\hat f_\text{val}^*$ on test data: $E_\text{test}$

Single metric

- Use RMS (Root Mean Squared) of train and dev error estimate to compare models
  - Harmonic mean not applicable as it gives more weight to smaller value

![image-20240627171239687](./assets/image-20240627171239687.png)

## Sampling Types

Repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model.

Hence, these help address the issue of a simple validation: Results can be highly variable, depending on which observations are included in the training set and which are in the validation set

|                  | Sampling        | Comment                                            | Better for identifying uncertainty in model |
| ---------------- | --------------- | -------------------------------------------------- | ------------------------------------------- |
| Bootstrapping    | w/ Replacement  | Better as we can have a large repetitions of folds | parameters                                  |
| Cross Validation | w/o Replacement |                                                    | accuracy                                    |

### Cross Validation Types

|                                  |                                                              | Purpose                                                      | Comment                                   |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| Regular $k$ fold                 | ![img](./assets/1*1L9DQtU1b7AHp8Bk4vr6QQ.png)                | Obtain uncertainty of evaluation estimates                   | Higher $k$ recommended for small datasets |
| Leave-One-Out                    |                                                              | For very small datasets<br />$n < 20$                        | $k=n$                                     |
| Shuffled                         | ![img](./assets/1*SUZHoDzHqYobxgtIZxRjrQ.png)                |                                                              |                                           |
| Random Permutation               | ![img](./assets/1*DD6nnhiXEpCZK406avyn6Q.png)                |                                                              |                                           |
| Stratified                       | ![](./assets/1*_Q298WoataU8CAhytaRAZA.png)                   | Ensures that Train, Validation & Test sets have same distribution |                                           |
| Stratified Shuffle               | ![img](./assets/1*c0fZiiLBNxHtw7-X7PJSGA.png)                |                                                              |                                           |
| Grouped                          | ![img](./assets/1*c1-TwkOsV_SwQWa-6qzKAg.png)<br />![](./assets/1*9v76vZK6lThlz0iehGd3vw.png) |                                                              |                                           |
| Grouped - Leave One Group Out    | ![img](./assets/1*sbITCGXoQEHSSm0xfjLD1A.png)                |                                                              |                                           |
| Grouped with Random Permutation  | ![img](./assets/1*oTIxF_Pjp13Y22ykDLLyKA.png)                |                                                              |                                           |
| Walk-Forward Expanding Window    | ![image-20240312120935236](./assets/image-20240312120935236.png)<br />![img](./assets/1*eVzEdnVo9BBAU33yM04ylA.png) |                                                              |                                           |
| Walk-Forward Rolling Window      | ![image-20240312120950200](./assets/image-20240312120950200.png) |                                                              |                                           |
| Blocking                         | ![img](./assets/1*MZ984wDUzAfWk6N8NNs9Rg.png)                |                                                              |                                           |
| Purging                          | ![img](./assets/1*rmkHSTJghRCa45lWCf9H_A.png)<br />![purged_cv](./assets/purged_cv.png)<br />![image-20240312120912110](./assets/image-20240312120912110.png) | Remove train obs whose labels overlap in time with test labels |                                           |
| Purging & Embargo                | ![img](./assets/1*a6x2YNDJkrwZD8VfV-YyPA.png)                | Prevent data leakage due to serial correlation $x_{\text{train}_{-1}} \approx x_{\text{test}_{0}}$<br />$y_{\text{train}_{-1}} \approx y_{\text{test}_{0}}$ |                                           |
| CPCV<br />(Combinatorial Purged) | ![image-20240312121125929](./assets/image-20240312121125929.png) |                                                              |                                           |

### Bootstrapping Types

|                                  |                |                                                              |
| -------------------------------- | -------------- | ------------------------------------------------------------ |
| Random sampling with replacement | IID            |                                                              |
| ARIMA Bootstrap                  | Parametric     |                                                              |
| Moving Block Bootstrap           | Non-parametric | ![image-20240312121539820](./assets/image-20240312121539820.png) |
| Circular Block Bootstrap         | Non-parametric |                                                              |
| Stationary Bootstrap             | Non-parametric |                                                              |

## Validation Methods

==Make sure to **shuffle** all splits for cross-sectional data==

| Type                     |                       Cross-Sectional                        |                         Time Series                          | Comment                                                      |
| ------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Holdout                  |      ![train_test_split](./assets/train_test_split.svg)      |      ![train_test_split](./assets/train_test_split.svg)      |                                                              |
| $k$- Fold                | ![k_fold_cross_validation](./assets/k_fold_cross_validation_cross_sectional_data.svg) | ![k_fold_cross_validation](./assets/k_fold_cross_validation_time_series_data.svg) | 1. Split dataset into $k$ subsets<br/>2. Train model on $(k-1)$ subsets<br />3. Evaluate performance on $1$ subset<br/>4. Summary stats of all iterations |
| Repeated $k$-Fold        | ![repeated_k_fold_cross_validation](./assets/repeated_k_fold_cross_validation_cross_sectional_data.svg) |                              ❌                               | Repeat $k$ fold with different splits and random seed        |
| Nested $k$-Fold          | ![nested_k_fold_cross_validation](./assets/nested_k_fold_cross_sectional_data.svg) | ![nested_k_fold_cross_validation](./assets/nested_k_fold_time_series_data.svg) |                                                              |
| Nested Repeated $k$-Fold | ![nested_repeated_k_fold_cross_validation](./assets/nested_repeated_k_fold_cross_sectional_data.svg) |                              ❌                               |                                                              |

### Decision Parameter $k$

There is a tradeoff

|            | Small $k$ | Large $k$ |
| ---------- | --------- | --------- |
| Train Size | Small     | Large     |
| Test Size  | Large     | Small     |
| Bias       | High      | Low       |
| Variance   | Low       | High      |

Usually $k$ is taken

- Large dataset: 4
- Small dataset: 10
- Tiny dataset: $k=n$ , ie LOOCV (Leave-One-Out CV)

![image-20240627180121829](./assets/image-20240627180121829.png)

## Data Leakage

Cases where some information from the training set has “leaked” into the validation/test set. Estimation of the performances is likely to be optimistic

Due to data leakage, model trained for $y_t = f(x_j)$ is more likely to be ‘luckily’ accurate, even if $x_j$ is irrelevant

Causes

- Perform feature selection using the whole dataset
- Perform dimensionality reduction using the whole dataset
- Perform parameter selection using the whole dataset
- Perform model or architecture search using the whole dataset
- Report the performance obtained on the validation set that was used to decide when to stop training (in deep learning)
- For a given patient, put some of its visits in the training set and some in the validation set
- For a given 3D medical image, put some 2D slices in the train- ing set and some in the validation set

