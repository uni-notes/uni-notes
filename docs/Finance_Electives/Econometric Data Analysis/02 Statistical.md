# Statistical Modelling

We try to come up with a hypothesis function $\hat y = h(x)$ which minimizes the cost function. The ‘best’ hypothesis function $h$ depends on the what kind of error we are trying to minimize.

|                |              Mean Squared<br />MSE              | Absolute<br />MAE |                     Indicator                      |
| -------------- | :---------------------------------------------: | :---------------: | :------------------------------------------------: |
| cost function  |                $(\hat y - y)^2$                 | $| \hat y - y |$  | $\begin{cases} 1, y \ne y \\ 0, y = y \end{cases}$ |
| penalizes      |                   large error                   |                   |                All inaccurate guess                |
| minimization   | easier<br />(will be taught in future lectures) |                   |                                                    |
| best statistic |                      Mean                       |      Median       |                        Mode                        |

The goal of the model isn’t to fit the existing data the best. The goal is to make the best predictions. Hence, we shouldn’t get tempted by overfitting.