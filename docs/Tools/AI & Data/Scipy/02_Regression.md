# Regression

```python
def linearModelLossRSS(b, X, y):
    # Make predictions
    predY = linearModelPredict(b, X)
    # Compute residuals
    res = y - predY
    # Compute the residual sum of squares
    residual_sum_of_squares = sum(res**2)
    # Compute the gradient of the loss
    gradient = -2 * np.dot(res, X)
    return (residual_sum_of_squares, gradient)
```

```python
import scipy.optimize as so

def linearModelFit(X, y, lossfcn):
    nrows, ncols = X.shape
    betas = np.zeros((ncols, 1))
    # Optimize the loss
    RES = so.minimize(
      lossfcn,
      betas,
      args=(X, y),
      jac=True,
      # hess = 2 # isn't it just 2
    )
    # Obtain estimates from the optimizer
    estimated_betas = RES.x
    # Compute goodness of fit
    res = y - np.mean(y)
    TSS = sum(res**2)
    RSS, deriv = linearModelLossRSS(estimated_betas, X, y)  # L2 loss and RSS are the same thing
    R2 = 1 - RSS / TSS
    return (estimated_betas, R2)
```

```python
X = np.array([[1, 0], [1, -1], [1, 2]])
y = np.array([0, 0.4, 2]) 

beta, R2 = linearModelFit(X, y, linearModelLossRSS)

print("Betas are", beta)
print("R2:\n", R2)
```

### Obtaining Jac and Hessian

```python
## Basic Linear Regression
x = sp.Symbol('x', constant=True, real=True)
y = sp.Symbol('y', constant=True, real=True)
m = sp.Symbol('m', real=True)
c = sp.Symbol('c', real=True)


yhat = m * x + c
l = (yhat - y)**2


vars = [m, b]


## First-Order Reaction


c0 = sp.Symbol('c_0', constant=True, real=True)
t = sp.Symbol('t', constant=True, real=True)
ct = sp.Symbol('c_t', constant=True, real=True)


k = sp.Symbol('k', real=True)


chat = c0 * sp.exp(-k*t)
l = (chat - ct)**2


vars = [k]
```

```python
## Printing

print("Function")
display(yhat)


print("\nLoss")
display(l)


print("\nJac")
for var in vars:
 display(l.diff(var).factor())


print("\nHess")
for var in vars:
 for var_inner in vars:
   display(l.diff(var).diff(var).factor())

```

