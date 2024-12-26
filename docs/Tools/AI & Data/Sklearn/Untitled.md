# Similar Libraries

## Boosted Trees

XgBoost, LightGBM, CatBoost

### Custom Loss Function

```python
# create model instance
from functools import partial

# custom_loss = torch.nn.MSELoss(reduction="sum") # sum is required; mean does not work
def custom_loss(y_pred, y_true):
  return (
      torch.abs(
	      ((y_pred-y_true)/y_true)
      )
      .sum() # sum is required; mean does not work
  )

custom_loss_grad_hess = partial(torch_autodiff_grad_hess, custom_loss)

model = XGBRegressor(
	objective = custom_loss_grad_hess,
)
# fit model
model.fit(X_train, y_train)
# make predictions
pred = model.predict(X_test)
```

#### Autograd

```python
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