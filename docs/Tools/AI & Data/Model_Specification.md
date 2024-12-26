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
class LogCosh(math.Math):
    def cost(self, true, pred, sample_weight, delta=None):
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
    def __init__(self, to_be_grouped=False):
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
        
        self.fitted_coeff_ = None
        self.fitted_coeff_formatted_ = None
        self.to_be_grouped = to_be_grouped

    def set_fitted_coeff(self, *fitted_coeff):
        self.fitted_coeff_ = fitted_coeff
    def set_fitted_coeff_formatted(self, *fitted_coeff_formatted):
        self.fitted_coeff_formatted_ = fitted_coeff_formatted
    def __str__(self):
        fillers = (
            self.args
            if self.fitted_coeff_formatted_ is None
            else self.fitted_coeff_formatted_
        )
        
        fillers = dict()
        
        if self.fitted_coeff_formatted_ is None:
            a, b = self.args, self.args
        else:
            a, b = self.args, self.fitted_coeff_formatted_
    
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
    def __init__(self):
        super().__init__(to_be_grouped=True)
    # @jax.jit
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
            ca # np.inf
        )
    def quantile(self, X_test, X_train, y_train, link_distribution_dof, link_distribution_q):
        ca_train = X_train["Previous_Reading"]
        t_train = X_train["Time_Point_Diff"]

        ct_train_true = y_train
        k_train_true = -(1/t_train) * (ct_train_true - ca_train)

        ct_train_hat = ( # chat_t
            self.equation(X_train, *self.fitted_coeff_)
        )
        k_train_hat = -(1/t_train) * (ct_train_hat - ca_train)

        u = k_train_hat - k_train_true # np.log(k_train_hat/k_train_true) # -(1/t_train) * np.log(ct_train_hat / ct_train_true) 
        
        X_train["u"] = u
        u_grouped = X_train.groupby("Temperature")["u"].agg(["mean", "std", "count"])
        u_grouped = u_grouped.reset_index()
        n = u_grouped["count"]
        u_grouped["se"] = u_grouped["std"] * np.sqrt(
            1 + 1/n
            # +
            # ((u_grouped["Temperature"] - u_grouped["Temperature"].mean())**2)/((n-1)*u_grouped["Temperature"].var()) # does not work for single temperature
        )
        u_grouped["u_quantile"] = scipy.stats.t.ppf(loc=0, scale=u_grouped["se"], df=link_distribution_dof, q = link_distribution_q)
        
        ca_test = X_test["Previous_Reading"]
        t_test = X_test["Time_Point_Diff"]
        
        grouped_last_time_point = X_train.groupby("Temperature")["Time_Point_Diff"].max().rename("Time_Point_Latest")
        
        ct_test_hat = ( # chat_t
            self.equation(X_test, *self.fitted_coeff_)
        )
        X_test["k_test_hat"] = -(1/t_test) * (ct_test_hat - ca_test)

        X_test = X_test.merge(u_grouped[["Temperature", "u_quantile"]], how="left", left_on="Temperature", right_on="Temperature")
        # X_test["k_test_quantile"] = np.exp(
        #     np.log(X_test["k_test_hat"]) - X_test["u_quantile"]
        # )
        X_test["k_test_quantile"] = X_test["k_test_hat"] - X_test["u_quantile"]
        X_test = X_test.merge(grouped_last_time_point, how="left", left_on="Temperature", right_on="Temperature")

        X_test["horizon"] = X_test["Time_Point_Diff"] - X_test["Time_Point_Latest"].max()
        X_test["horizon"] = np.clip(
            X_test["horizon"],
            0,
            np.inf
        )
        # - for correction

        response_quantile_pred_test = ca_test - (X_test["k_test_quantile"] * t_test)

        response_quantile_pred_test = np.clip(
            response_quantile_pred_test,
            0,
            np.inf
        )
        
        return response_quantile_pred_test

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
    def __init__(self):
        super().__init__(to_be_grouped=True)
    # @jax.jit
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
    def quantile(self, X_test, X_train, y_train, link_distribution_dof, link_distribution_q):
        ca_train = X_train["Previous_Reading"]
        t_train = X_train["Time_Point_Diff"]

        ct_train_true = y_train
        k_train_true = -(1/t_train) * np.log(ct_train_true / ca_train)

        ct_train_hat = ( # chat_t
            self.equation(X_train, *self.fitted_coeff_)
        )
        k_train_hat = -(1/t_train) * np.log(ct_train_hat / ca_train)

        u = np.log(k_train_hat/k_train_true) # -(1/t_train) * np.log(ct_train_hat / ct_train_true) 
        
        X_train["u"] = u
        u_grouped = X_train.groupby("Temperature")["u"].agg(["mean", "std", "count"])
        u_grouped = u_grouped.reset_index()
        n = u_grouped["count"]
        u_grouped["se"] = u_grouped["std"] * np.sqrt(
            1 + 1/n
            # +
            # ((u_grouped["Temperature"] - u_grouped["Temperature"].mean())**2)/((n-1)*u_grouped["Temperature"].var()) # does not work for single temperature
        )
        u_grouped["u_quantile"] = scipy.stats.t.ppf(loc=u_grouped["mean"], scale=u_grouped["se"], df=link_distribution_dof, q = link_distribution_q)
        
        ca_test = X_test["Previous_Reading"]
        t_test = X_test["Time_Point_Diff"]
        
        grouped_last_time_point = X_train.groupby("Temperature")["Time_Point_Diff"].max().rename("Time_Point_Latest")
        
        ct_hat = ( # chat_t
            self.equation(X_test, *self.fitted_coeff_)
        )
        X_test["k_test_hat"] = -(1/t_test) * np.log(ct_hat / ca_test)

        X_test = X_test.merge(u_grouped[["Temperature", "u_quantile"]], how="left", left_on="Temperature", right_on="Temperature")
        # X_test["k_test_quantile"] = np.exp(
        #     np.log(X_test["k_test_hat"]) - X_test["u_quantile"]
        # )
        X_test["k_test_quantile"] = X_test["k_test_hat"] / np.exp(X_test["u_quantile"])
        X_test = X_test.merge(grouped_last_time_point, how="left", left_on="Temperature", right_on="Temperature")

        X_test["horizon"] = X_test["Time_Point_Diff"] - X_test["Time_Point_Latest"].max()

        X_test["horizon"] = np.clip(
            X_test["horizon"],
            0,
            np.inf
        )
        
        # - for correction
        response_quantile_pred_test = ca_test * np.exp(-(X_test["k_test_quantile"] * t_test))

        response_quantile_pred_test = np.where(
            response_quantile_pred_test.round(2) == 0.0,
            response_quantile_pred_test + link_distribution_q,
            response_quantile_pred_test
        )
        # st.write(response_quantile_pred_test)
        
        return response_quantile_pred_test

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

class Combined_Arrhenius(Model):
    # @jax.jit
    def equation(
        self, x,
        A=[0, (0, None)],
        E_a=[10_000, (0, None)],
        n=[1, (0.8, 1.2)],
    ):
        ca = x["Previous_Reading"]
        # ta = x["Time_Point_Diff"]
        # tb = x["Time_Point"]
        t = x["Time_Point_Diff"]
        T = x["Temperature"] + 273.15
        # Conversion to Kelvin is required: Kelvin is a ratio attribute, while Celcius and Farenheit are not
        # Please read https://uni-notes.netlify.app/CS_Electives/Data_Mining/02_Data/#types-of-attributes for more details
        R = 8.3144598
        
        k = (
            A
            *
            np.exp(
                -E_a
                /
                (R*T).values
            )
        )

        return (
            ca
            *
            np.exp(
                -k * (t**n) # (tb-ta)
            )
        )

    def quantile(self, X_test, X_train, y_train, link_distribution_dof, link_distribution_q):
        ca_train = X_train["Previous_Reading"]
        t_train = X_train["Time_Point_Diff"]

        ct_train_true = y_train
        k_train_true = -(1/t_train) * np.log(ct_train_true / ca_train)

        ct_train_hat = ( # chat_t
            self.equation(X_train, *self.fitted_coeff_)
        )
        k_train_hat = -(1/t_train) * np.log(ct_train_hat / ca_train)

        u = np.log(k_train_hat/k_train_true) # -(1/t_train) * np.log(ct_train_hat / ct_train_true) 
        
        X_train["u"] = u
        u_grouped = X_train.groupby("Temperature")["u"].agg(["mean", "std", "count"])
        u_grouped = u_grouped.reset_index()
        n = u_grouped["count"]
        u_grouped["se"] = u_grouped["std"] * np.sqrt(
            1 + 1/n
            # +
            # ((u_grouped["Temperature"] - u_grouped["Temperature"].mean())**2)/((n-1)*u_grouped["Temperature"].var()) # does not work for single temperature
        )
        u_grouped["u_quantile"] = scipy.stats.t.ppf(loc=u_grouped["mean"], scale=u_grouped["se"], df=link_distribution_dof, q = link_distribution_q)
        
        ca_test = X_test["Previous_Reading"]
        t_test = X_test["Time_Point_Diff"]
        
        grouped_last_time_point = X_train.groupby("Temperature")["Time_Point_Diff"].max().rename("Time_Point_Latest")
        
        ct_hat = ( # chat_t
            self.equation(X_test, *self.fitted_coeff_)
        )
        X_test["k_test_hat"] = -(1/t_test) * np.log(ct_hat / ca_test)

        X_test = X_test.merge(u_grouped[["Temperature", "u_quantile"]], how="left", left_on="Temperature", right_on="Temperature")
        # X_test["k_test_quantile"] = np.exp(
        #     np.log(X_test["k_test_hat"]) - X_test["u_quantile"]
        # )
        X_test["k_test_quantile"] = X_test["k_test_hat"] / np.exp(X_test["u_quantile"])
        X_test = X_test.merge(grouped_last_time_point, how="left", left_on="Temperature", right_on="Temperature")

        X_test["horizon"] = X_test["Time_Point_Diff"] - X_test["Time_Point_Latest"].max()
        X_test["horizon"] = np.clip(
            X_test["horizon"],
            0,
            np.inf
        )
        
        # - for correction

        response_quantile_pred_test = ca_test * np.exp(-(X_test["k_test_quantile"] * t_test))

        response_quantile_pred_test = np.where(
            response_quantile_pred_test.round(2) == 0.0,
            response_quantile_pred_test + link_distribution_q,
            response_quantile_pred_test
        )
        
        return response_quantile_pred_test

    def latex(self):
        return r"""
        $$
        \begin{aligned}
        {\huge c_t} &
        {\huge = c_0 \cdot \exp \Bigg \{
            -
            \overset{
                \mathclap{
                \substack{ {\small \text{Rate const}} \qquad \\  {\tiny \nwarrow} \qquad \\ }
                }
            }{
                k
            }
            
            t^ {
                \overset{
                \mathclap{\qquad
                {\tiny \nearrow} \substack{ {\small \text{Order}} \\ \\ \\ }
                }
                }{
                \textcolor{hotpink}{$n}
                }
            }
        \Bigg \} } \\
        {\large k} &=
        {
            \large \textcolor{hotpink}{$A}
            \exp \left\{
            {
                \dfrac{
                    -\textcolor{hotpink}{$E_a}
                }{
                    RT
                }
            }
            \right\}
        } \\
        \\
        c_t &= \text{Concentration} \\
        c_0 &= \text{Initial Concentration} \\
        T &= \text{Temperature (Kelvin)} \\
        R &= \text{Gas Constant} \\
        t &= \text{Time (weeks)}
        \end{aligned}
        $$
        """
        
class Rate_Law(Model):
    # def __init__(self):
    #     super().__init__()
    #     self.constraints = (
    #         {'type': 'ineq', 'fun': lambda x:  x[-1] != 0}, # n!=0
    #     )
    #     #Nelder-Mead doesn't support constraints
        
    # @jax.jit
    def equation(
        self, x,
        A=[1, (0, None)],
        E_a=[10_000, (0, None)],
        n=[1, (0.8, 1.2)],
    ):
        ca = x["Previous_Reading"]
        # ta = x["Time_Point_Diff"]
        # tb = x["Time_Point"]
        t = x["Time_Point_Diff"]
        T = x["Temperature"] + 273.15
        # Conversion to Kelvin is required: Kelvin is a ratio attribute, while Celcius and Farenheit are not
        # Please read https://uni-notes.netlify.app/CS_Electives/Data_Mining/02_Data/#types-of-attributes for more details
        R = 8.3144598
        
        k = (
            A
            *
            np.exp(
                -E_a
                /
                (R*T)
            )
        )

        return np.power(
            ca**(1-n) - k*t*(1-n), # (tb-ta)
            1/(1-n)
        )

    def quantile(self, X_test, X_train, y_train, link_distribution_dof, link_distribution_q):
        ca_train = X_train["Previous_Reading"]
        t_train = X_train["Time_Point_Diff"]

        ct_train_true = y_train
        k_train_true = -(1/t_train) * np.log(ct_train_true / ca_train)

        ct_train_hat = ( # chat_t
            self.equation(X_train, *self.fitted_coeff_)
        )
        k_train_hat = -(1/t_train) * np.log(ct_train_hat / ca_train)

        u = np.log(k_train_hat/k_train_true) # -(1/t_train) * np.log(ct_train_hat / ct_train_true) 
        
        X_train["u"] = u
        u_grouped = X_train.groupby("Temperature")["u"].agg(["mean", "std", "count"])
        u_grouped = u_grouped.reset_index()
        n = u_grouped["count"]
        u_grouped["se"] = u_grouped["std"] * np.sqrt(
            1 + 1/n
            # +
            # ((u_grouped["Temperature"] - u_grouped["Temperature"].mean())**2)/((n-1)*u_grouped["Temperature"].var()) # does not work for single temperature
        )
        u_grouped["u_quantile"] = scipy.stats.t.ppf(loc=u_grouped["mean"], scale=u_grouped["se"], df=link_distribution_dof, q = link_distribution_q)
        
        ca_test = X_test["Previous_Reading"]
        t_test = X_test["Time_Point_Diff"]
        
        grouped_last_time_point = X_train.groupby("Temperature")["Time_Point_Diff"].max().rename("Time_Point_Latest")
        
        ct_hat = ( # chat_t
            self.equation(X_test, *self.fitted_coeff_)
        )
        X_test["k_test_hat"] = -(1/t_test) * np.log(ct_hat / ca_test)

        X_test = X_test.merge(u_grouped[["Temperature", "u_quantile"]], how="left", left_on="Temperature", right_on="Temperature")
        # X_test["k_test_quantile"] = np.exp(
        #     np.log(X_test["k_test_hat"]) - X_test["u_quantile"]
        # )
        X_test["k_test_quantile"] = X_test["k_test_hat"] / np.exp(X_test["u_quantile"])
        X_test = X_test.merge(grouped_last_time_point, how="left", left_on="Temperature", right_on="Temperature")

        X_test["horizon"] = X_test["Time_Point_Diff"] - X_test["Time_Point_Latest"].max()
        X_test["horizon"] = np.clip(
            X_test["horizon"],
            0,
            np.inf
        )
        
        # - for correction

        response_quantile_pred_test = ca_test * np.exp(-(X_test["k_test_quantile"] * t_test))

        response_quantile_pred_test = np.where(
            response_quantile_pred_test.round(2) == 0.0,
            response_quantile_pred_test + link_distribution_q,
            response_quantile_pred_test
        )
        
        return response_quantile_pred_test

    def latex(self):
        # {\huge c_t} & {\huge = c_0 \cdot e^{
        return r"""
        $$
        \begin{aligned}
        {\large c_t^{1-\textcolor{hotpink}{$n}}}
        &=
        {
            \Large
            c_0^{1-\textcolor{hotpink}{$n}}
            - k t
            ( {1 - \textcolor{hotpink}{$n}} )
        } \\
        {\large k} &=
        {
            \large \textcolor{hotpink}{$A}
            \exp \left\{
            {
                \dfrac{
                    -\textcolor{hotpink}{$E_a}
                }{
                    RT
                }
            }
            \right\}
        } \\
        \\
        c_t &= \text{Concentration} \\
        c_0 &= \text{Initial Concentration} \\
        T &= \text{Temperature (Kelvin)} \\
        R &= \text{Gas Constant} \\
        t &= \text{Time (weeks)}
        \end{aligned}
        $$
        """
```
