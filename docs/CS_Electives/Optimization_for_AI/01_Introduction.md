# Optimization

Find a set of parameters that minimizes the loss function for the given data and algorithm
$$
\underset{\theta}{\arg \min} \ L( \ y, \hat f_\theta(D) \ )
$$

## IDK

Always test optimization procedure on known solution

## Backpropagation Steps

1. Forward pass
2. Calculate loss 
3. Backward pass

Basically just chain rule + intelligent caching of intermediate results

Computationally-cheap, but large storage required for caching intermediate results

Each layer needs to be able to perform vector Jacobian product: multiply the “incoming backward gradient” by its derivatives
$$
\begin{aligned}
\dfrac{\partial J}{\partial \theta_l}
&=
\dfrac{\partial J}{\partial y_L}
\left(
\prod_{i=l}^L
\dfrac{\partial y_i}{\partial y_{i-1}}
\right)
\dfrac{\partial y_l}{\partial \theta_l}
\end{aligned}
$$
![image-20240710181826692](./assets/image-20240710181826692.png)

## Training Process

```mermaid
flowchart LR
i[Initialize θ] -->
j[Calculate cost function] -->
a{Acceptable?} -->
|Yes| stop([Stop])

a -->
|No| change[Update θ] -->
j
```

### Steps

- Forward pass
- Backward pass
- Weights update

## Optimization Parameters

### Objective Function

An objective function has a unique global minimum if it is

- differentiable
- convex

![image-20240216005549032](./assets/image-20240216005549032.png)

### Hard Constraints and Bounds

Useful if you know the underlying systematic differential equation

$$
\text{DE} = 0
$$
Refer to PINNs for more information

When it is not possible to use a discontinuous hard constraint/bound (such as $\beta \ge k$), you can add a barrier function to the cost function
$$
J' = J + B
$$
where $B$ can be

- Exponential barrier: $\pm \exp \{ m (\beta - k) \}$
  - where $m=$ barrier coefficient

![image-20240413120123186](./assets/image-20240413120123186.png)

### Soft Constraints: Regularization

Encourages (not guaranteed) certain parameters to end in range values, through penalizing deviation from prior/preferred values

## Weights Initialization Algorithm

- Zero (bad)
- Random
- Glorot (Xavier)

## Optimization Algorithms

[Optimization Algorithms](./../Optimization_Algorithms)

## Batch Size

| Optimizer                               | Meaning                                                      | Comment                                                      | Gradient-Free | Weight Update Rule<br />$w_{t+1}$ | Advantages                                                   | Disadvantages                                                |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- | --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BGD<br />(Batch Gradient Descent)       | Update weights after viewing the entire dataset: $n$ sample points |                                                              | ❌             |                                   | Guaranteed convergence to local minimum                      | Computationally-expensive for large dataset<br />Prone to getting stuck at non-optimal local minima for non-convex cost functions |
| SGD<br />(Stochastic Gradient Descent)  | Update weights after viewing every sample point              |                                                              | ❌             | $w_t - \eta g(w_t)$               | Cheaper computation<br />Faster updates<br />Randomization helps escape ‘shallow’ local minima | May not converge to global minima for non-convex cost functions<br />Noisy/Oscillating/Erratic convergence |
| MBGD<br />(Mini-Batch Gradient Descent) | Update weights after viewing the $b$ sample points, where $b < n$<br /><br />Usually $b=32$ | Middle ground between BGD and SGD<br />Generalizes better than Adam | ❌             |                                   |                                                              |                                                              |

Rule of thumb for SGD: recommended $\eta = 0.1$

## Gradient Descent

Similar to trial and error

1. Start with some $\theta$ vector
2. Keep changing $\theta_0, \theta_1, \dots, \theta_n$ using derivative of cost function, until minimum for $J(\theta)$ is obtained - **Simultaneously**

$$
\theta_{\text{new}} =
\theta_{\text{prev}} -
\eta \ 
{\nabla J}
$$

|                       | Meaning                                                      |
| --------------------- | ------------------------------------------------------------ |
| $\theta_{\text{new}}$ | Coefficients obtained from current iteration<br />(Output of current iteration) |
| $\theta_{\text{old}}$ | Coefficients obtained from previous iteration<br />(Output of previous iteration) |
| $\eta$                | Learning Rate                                                |
| $\nabla J$            | Gradient vector of $J (\theta)$                              |

### Gradients of the Loss Function

![image-20240704165816976](./assets/image-20240704165816976.png)

![image-20240704170135799](./assets/image-20240704170135799.png)

### Learning Rate $\eta$

$0 < \eta < 1$

- Large value may lead to underfitting/overfitting
- Small value will lead to more time taken

Can be

- constant
- time-based decay

![image-20240216010116161](./assets/image-20240216010116161.png)

## Iterative vs Normal Equation

|                              |            Iterative            |              Normal Equation               |
| :--------------------------: | :-----------------------------: | :----------------------------------------: |
|  $\alpha$ **not** required   |                ❌                |                     ✅                      |
| Feature scaling not required |                ❌                |                     ✅                      |
|       Time Complexity        |            $O(kn^2)$            |                  $O(n^3)$                  |
|         Performance          |     Fast even for large $n$     |             Slow if $n > 10^4$             |
|        Compatibility         |    Works for all algorithms     |      Doesn’t work for classification       |
|        No of features        |    Works for all algorithms     | Doesn't work when $X^TX$ is non-invertible |
|        Stop criteria         |                                 |                    None                    |
|         Convergence          |              Slow               |                                            |
|  Global Optimal guaranteed   |                ❌                |                     ✅                      |
|        Loss Function         | Should be double-differentiable |                                            |

Gradient-based methods find min of a function by moving in the direction in which the function decreases most steeply

## Speed Up Training

- Subsetting
- Feature-scaling
- Pruning
- Good Weight initialization
- Good Activation functions
- Transfer learning: Re-use parts of pre-trained network
- Using mini-batch updates
- Learning rate scheduling
- Faster optimization algorithm
- Use GPU/TPU

### Subsetting

1. Sample Size
   - Mini-Batch
   - Stochastic
2. Input Features

You can do either

- drop with both approaches
- Bagging with each sub-model using the subset

### Feature Scaling

Helps to speed up gradient descent by making it easier for the algorithm to reach minimum faster

Get every feature to approx $-1 \le x_i \le 1$ range

Atleast try to get $-3 \le x_i \le 3$ or $-\frac13 \le x_i \le \frac13$

#### Standardization

$$
\begin{aligned}
x'_i
&= z_i \\
&= \frac{ x_i - \bar x }{s}
\end{aligned}
$$

#### Batch Normalization

![img](./assets/1*vXpodxSx-nslMSpOELhovg.png)

## Learning Rate

![img](./assets/1*rcmvCjQvsxrJi8Y4HpGcCw.png)

## Convex Function

Convex function is one where
$$
\begin{aligned}
f(\alpha x + \beta y) &\le \alpha f(x) + \beta f(y) \\
\alpha + \beta &= 1; \alpha, \beta \ge 0
\end{aligned}
$$

## Robust Optimization

![image-20240211220341621](./assets/image-20240211220341621.png)

### Limitations

- Parameters must be independent
- Cannot handle equality constraints
- Hard to estimate min and max value of parameter
- Method is extremely conservative

## Batch Size

### Resources

Let $b=$ batch size
$$
\begin{aligned}
\text{Space Requirement} &\propto \dfrac{1}{b} \\
\text{Time Requirement} &\propto b
\end{aligned}
$$

- Larger batch size means larger memory required to train a single batch at one time
- Larger batch size means fewer updates per epoch

### Generalization

The following is only empirically-proven
$$
\begin{aligned}
\text{Generalization} &\propto \dfrac{1}{b}
\end{aligned}
$$
The noise from smaller batch size helps escape suboptimal local minimum

![img](./assets/1*5mHkZw3FpuR2hBNFlRxZ-A.png)

![img](./assets/1*PV-fcUsNlD9EgTIc61h-Ig.png)

### Learning Rate

Should be scaled according to batch size
$$
\text{LR}' = \text{LR} \times (b/32)
$$

## Batching

When training a neural network, we usually divide our data in mini-batches and go through them one by one. The network predicts batch labels, which are used to compute the loss with respect to the actual targets. Next, we perform backward pass to compute gradients and update model weights in the direction of those gradients.

- Full dataset does not fit in memory
- Faster convergence due to stochasticity

## Approaches to obtain gradient

- Exact: Use matrix differential calculus, Jacobians, Kronecker products, & vectorization
- Approximation
  1. Pretend everything everything is a scalar
  2. use typical chain rule
  3. rearrange/transpose matrices/vectors to make the sizes work
  4. verify result numerically 

## Initialization

Initialization is very important: Weights don’t move “that much”, so weights tend often stay much closer to initial points than to the “final” point after optimization from different initial point

If you initialize all the weights as 0, all your gradients will be 0 and ANN will not learn anything
$$
\begin{aligned}
W_{t=0} &= N(0, \sigma^2 I) \\
\sigma^2_\text{recom RELU} &= \dfrac{2}{\text{no of neurons}} \\
\sigma^2_{\text{recom } \sigma} &= \dfrac{1}{\text{no of neurons}}
\end{aligned}
$$
Kaiming Normal Initialization: based on central limit theorem, we want the entire distribution to become $N(0, 1)$

The choice of $\sigma^2$ will affect

1. Magnitude/Norm of forward activations
2. Magnitude/Norm of gradients

| $\sigma^2$ | Norms     |
| ---------- | --------- |
| Too low    | Vanishing |
| Optimal    | Right     |
| Too high   | Exploding |

![image-20240525155739867](./assets/image-20240525155739867.png)

Here $n=$ no of neurons

Why is $\sigma^2 = 2/n$ the best? Because ReLU will cause half the components of the activations to be set to 0, so we need twice the variance to achieve the same final variance

Even when trained successfully, the effects/scales present at initialization persist throughout training

![image-20240525163200403](./assets/image-20240525163200403.png)

### Solution

|           | Layer Normalization                                                                              | Batch Normalization | |
|---        | ---                                                                                              | ---                 | ---|
| | Normalize activations of each image at each layer | Normalize activations of all images in each mini-batch at each layer | |
|$w'_{i+1}$ | $\dfrac{w_{i+1} - E[w_{i+1}]}{\sigma(w_{i+1}) + \epsilon}$                                      |                     | |
| | ![image-20240525165138330](./assets/image-20240525165138330.png) | ![image-20240525165203016](./assets/image-20240525165203016.png) | |
|           | ![image-20240525164536090](./assets/image-20240525164536090.png)                                 |                     | |
|Limitation | Harder to train standard FCN to low loss, because the relative sizes between activations is lost | Inter-dependence of training samples (Soln: below) | |

Where

- $i=$ layer number
- $\epsilon={10}^{-5}$

### Soln

1. Training: Compute running average of mean $\hat \mu_{i+1}$ & variance $\hat \sigma^2_{i+1}$ for all features at each layer
2. Inference: Normalize by these quantities

$$
(w'_{i+1})_j = \dfrac{(w_{i+1})_j - (\hat \mu_{i+1})_j}{(\hat \sigma_{i+1})_j + \epsilon}
$$

## Stopping Criteria

Use an `or` combination of the following

- $J(\theta) \le$  Cost Threshold
- $\vert J(\theta)_e - J(\theta)_{e-1} \vert \le$ Convergence threshold
  - where $e=$ epochs
  - Moving average of the previous 5?

- Evaluation metric $\le$ Evaluation threshold
  - This may be different from the cost function
  - MSE for cost; MAPE for evaluation
- $n_{\text{iter}} \ge$ Iter Threshold
- Time taken $\ge$  Duration threshold

## IDK

For each epoch, you can subsample the training set and then create batches

- Cheaper epochs
- More stochastic
