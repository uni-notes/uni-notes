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
class Huber(Math):
  def cost(self, pred, true, sample_weight, delta=None):
    error = pred - true
    error_abs = np.abs(error)

    if delta is None:
      ## delta = 0.1
      delta = 1.345 * np.std(error)

    ## huber loss
    loss = np.where(
      error_abs > delta,
      (error**2 / 2),
      (
        delta * error_abs -
        delta**2 / 2
      )
    )

    cost = np.mean(
      sample_weight * loss
    )
    return cost

  def latex(self):
    return r"""
    $$
    \Large
    \text{mean}
    \begin{cases}
    \dfrac{u_i^2}{2}, & \vert u_i \vert > \delta \\
    \delta \vert u_i \vert - \dfrac{\delta^2}{2}, & \text{otherwise}
    \end{cases} \\
    \delta_\text{recommended} = 1.345 \sigma_u
    $$
    """
```
## Models
```python
class Model(Math):
  def __init__(self):
    args = getfullargspec(self.equation).args
    self.args = tuple([arg for arg in args if arg not in ["self", "x"]])

    self.defaults = getfullargspec(self.equation).defaults

    self.initial_guess = (
      [0 for arg in self.args]
      if (self.defaults is None) or (len(self.args) != len(self.defaults))
      else self.defaults
    )

    self.fitted_coeff = None

  def set_fitted_coeff(self, *fitted_coeff):
    self.fitted_coeff = fitted_coeff
  
  def __str__(self):
    fillers = (
      self.args
      if self.fitted_coeff is None
      else self.fitted_coeff
    )

    return self.latex() % fillers
```
```python
class Arrhenius(Model):
  def equation(self, x, k):  ## hypothesis
    i = x["Initial_Reading"]
    t = x["Time_Point"]
    return i * np.exp(-1 * k * t)
  def latex(self):
    return r"""
    $$
    \begin{aligned}
    {\huge c_t} & {\huge = c_0 \cdot e^{
      \overbrace{\textcolor{hotpink}{-%s}}^{\small \mathclap{\small \mathclap{\text{Rate Constant}}}} t
    } } & \text{(Arrhenius Eqn)} \\
    \text{where }
    c_t &= \text{Concentration} \\
    c_0 &= \text{Initial Concentration} \\
    t &= \text{Time (weeks)}
    \end{aligned}
    $$
    """
```
```python
class Exponential_Combined_Model(Model):
  def equation(self, x, a=0, b=0, n=1):
    i = x["Initial_Reading"]
    t = x["Time_Point"]
    T = x["Temperature"]

    return i * np.exp(
      a * np.exp(b * T) * (t**n)
    )

  def latex(self):
    return r"""
    $$
    \begin{aligned}
    {\huge c_t} & {
    \huge = c_0 \cdot e^{
      \overbrace{\textcolor{hotpink}{%s} (e^{
      \footnotesize \textcolor{hotpink}{%s} T} )}^{\mathclap{\small \text{Rate Constant}}} \ t^ {
        \overset{
        \mathclap{\quad \quad \quad \quad
        {\tiny \nearrow} \ \substack{ {\small \text{Order of Rxn}} \\ \\ }
        }
        }{
        \textcolor{hotpink}{%s}
        }
      }
    }
    } \\
    \text{where }
    c_t &= \text{Concentration} \\
    c_0 &= \text{Initial Concentration} \\
    T &= \text{Temperature} \\
    t &= \text{Time (weeks)}
    \end{aligned}
    $$
    """
```