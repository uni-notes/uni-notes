# Model Specification

## Base

```python
class Math:
    def __str__(self):
        return self.latex()
    def __repr__(self):
        return str(self)
    def equation(self):
        return ""
    def latex(self):
        return ""
```
## Cost Functions
```python
class LogCosh(Math):
    def cost(self, pred, true, sample_weight, delta=None):
        error = pred - true

        loss = np.log(np.cosh(error))
        
        cost = np.mean(
            sample_weight * loss
        )
        return cost
    
    def latex(self):
        return r"""
        $$
        \Large
        \text{Mean Log Cosh}: \text{mean} \Big\{ \log \left \vert \ \cosh (u_i) \ \right \vert \Big \} \\
        \text{where } u_i = \text{ Prediction - True}
        $$
        """
```
## Models

```python
from utils.math import *

from inspect import getfullargspec, getsource
from string import Template
```

```python
class Model(Math):
    def __init__(self):
        name = (
            self.__class__.__name__
            .replace("_", " ")
            .replace("Model", "")
            .strip()
        )
        self.__class__.__name__ = name
        self.__name__ = name
        
        args = getfullargspec(self.equation).args
        self.args = tuple([arg for arg in args if arg not in ["self", "x"]])
        self.k = len(self.args)
        
        self.defaults = list(getfullargspec(self.equation).defaults)
        
        self.param_initial_guess = [
            x[0]
            for x
            in self.defaults
        ]
        
        self.param_initial_guess = (
            [0 for arg in self.args]
            if (self.param_initial_guess is None) or (self.k != len(self.param_initial_guess))
            else self.param_initial_guess
        )
        
        self.param_bounds = [
            x[1]
            for x
            in self.defaults
        ]
        
        self.param_bounds = (
            [(None, None) for arg in self.args]
            if (self.param_bounds is None) or (self.k != len(self.param_bounds))
            else self.param_bounds
        )
        
        if "constraints" not in dir(self):
            self.constraints = []
        
        self.fitted_coeff = None

    def set_fitted_coeff(self, *fitted_coeff):
        self.fitted_coeff = fitted_coeff
    def __str__(self):
        fillers = (
            self.args
            if self.fitted_coeff is None
            else self.fitted_coeff
        )
        
        fillers = dict()
        
        if self.fitted_coeff is None:
            a, b = self.args, self.args
        else:
            a, b = self.args, self.fitted_coeff
    
        for key, value in zip(a, b):
            fillers[key] = value
            
        equation = Template(self.latex()).substitute(fillers).replace("$", "$$")
        # st.code(string)
              
        return rf"""
        $$
        \text{{ {self.__class__.__name__} }}
        $$
        
        {equation}
        """
```
### Example

```python
class Zero_Order(Model):
    def equation(
        self, x,
        k = [0, (0, None)]
    ):  # hypothesis
        ca = x["Previous_Reading"]
        # ta = x["Time_Point_Diff"]
        # tb = x["Time_Point"]
        t = x["Time_Point_Diff"]
        return np.clip(
            (
                ca - k * t
            ),
            0,
            np.inf
        )
    def latex(self):
        return r"""
        $$
        \begin{aligned}
        {\huge c_t} &
        {\huge = c_0 - \textcolor{hotpink}{$k} t} \\
        \\
        c_t &= \text{Concentration} \\
        c_0 &= \text{Initial Concentration} \\
        t &= \text{Time (weeks)}
        \end{aligned}
        $$
        """

class First_Order(Model):
    def equation(
        self, x,
        k = [0, (0, None)]
    ):  # hypothesis
        ca = x["Previous_Reading"]
        # ta = x["Time_Point_Diff"]
        # tb = x["Time_Point"]
        t = x["Time_Point_Diff"]
        return (
            ca
            *
            np.exp(
                -k
                *
                t # (tb-ta)
            )
        )
    def latex(self):
        return r"""
        $$
        \begin{aligned}
        {\huge c_t} &
        {\huge = c_0 \cdot
        \exp \{
            \overbrace{- \textcolor{hotpink}{$k}}^{\small \mathclap{\small \mathclap{\text{Rate const}}}} t
        \} } \\
        \\
        c_t &= \text{Concentration} \\
        c_0 &= \text{Initial Concentration} \\
        t &= \text{Time (weeks)}
        \end{aligned}
        $$
        """
```
