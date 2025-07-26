# Boosted Trees

XgBoost, LightGBM, CatBoost


```python
model = XGBRegressor(
	objective = custom_loss_grad_hess,
)
# fit model
model.fit(X_train, y_train)
# make predictions
pred = model.predict(X_test)
```

## Custom Loss Function

Need a function that returns
- `grad`: $\dfrac{dL}{d \hat y}$
- `hess`: $\dfrac{d^2 L}{d {\hat y}^2}$

```python
# define cost and eval functions
def custom_loss(y_pred, y_true):
    residual = (y_true - y_pred)
    loss = np.where(
	    residual < 0,
	    (residual ** 2) ,
	    (residual ** 2) * 2
	)
    return np.mean(loss)

def custom_loss_grad_hess(y_pred, y_true):
    residual = (y_true - y_pred)
    grad = np.where(
	    residual < 0,
	    (-2 * residual),
	    (-2 * residual) * 2
	)
    hess = np.where(
	    residual < 0,
	    2,
	    2 * 2
	)
    return grad, hess
```

## Undefined Hessian

MAE and MAPE don't work as the second derivative is 0, and no learning happens

### Option 1: Set hessian to 1


```python
# define cost and eval functions
def mae(y_pred, y_true):
    residual = (y_true - y_pred)
    loss = np.abs(residual)
    return np.mean(loss)

def mae_grad_hess(y_pred, y_true):
    #residual = (y_true - y_pred)
    grad = np.ones(y_pred.shape) * -2
    hess = np.ones(y_pred.shape)
    return grad, hess
```

### Option 2: Randomized version

```python
def quantile_loss(y_true,y_pred,alpha,delta,threshold,var):
    x = y_true - y_pred
    
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-  ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
 
    grad = (np.abs(x)<threshold )*grad - (np.abs(x)>=threshold )*(2*np.random.randint(2, size=len(y_true)) -1.0)*var
    hess = (np.abs(x)<threshold )*hess + (np.abs(x)>=threshold )
    
    return grad, hess
```

### Using sympy

```python

```

### Using autograd

```python
custom_loss_grad_hess = partial(torch_autodiff_grad_hess, custom_loss)

# create model instance
from functools import partial

# custom_loss = torch.nn.MSELoss(reduction="sum") # sum is required; mean does not work
def custom_loss(y_pred, y_true):
  return (
      (
	      (y_pred-y_true) /
	      y_true
      ) **2
      .sum() # sum is required; mean does not work
  )

def torch_autodiff_grad_hess(
    loss,
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    """Perform automatic differentiation to get the
    Gradient and the Hessian of `loss_function`.
    """
    
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    y_pred.requires_grad_()
    
    loss_lambda = lambda y_pred: loss(y_pred, y_true)

    grad = torch.autograd.functional.jacobian(
        loss_lambda,
        y_pred,
        vectorize = True,
    )

    hess_matrix = torch.autograd.functional.hessian(
        loss_lambda,
        y_pred,
        vectorize=True,
    )
    hess = torch.diagonal(hess_matrix)

    return grad.numpy(), hess.numpy()
```
