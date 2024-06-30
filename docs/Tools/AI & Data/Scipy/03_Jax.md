# Jax

```python
# from scipy.optimize import minimize

from jax.scipy.optimize import minimize
import jax.numpy as np
```

Gradients of fun are calculated automatically using JAX’s autodiff support when required.

## Speed Up Existing Implementation

```python
import jax.numpy as np
from jax import jit, value_and_grad

def costFunction(weights):
   # reshapes flattened weights into 2d matrix
   weights = np.reshape(weights, newshape=(100, 75))

   # weighted row-wise sum
   weighted = np.sum(x * weights, axis=1)

   # squared residuals
   residualsSquared = (y - weighted) ** 2

   return np.sum(residualsSquared)

# create the derivatives
obj_and_grad = jit(value_and_grad(costFunction))

minimize(obj_and_grad, x0=startingWeights.flatten(), jac=True)
```

Or take the diff using `jax.diff`

- Provide the jac & hessian to the minimize function

## Use `jax.scipy`

```python
import 
o.minimize(fun, x0, args=(), *, method, tol=None, options=None)
```

