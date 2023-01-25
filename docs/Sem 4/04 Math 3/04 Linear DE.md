## Linear DE General Form

$$
y' + P y = Q

$$

where $y$ is the dependent variable

### Solution

1. Find IF $= e^{\int P(x) dx}$

2. Find general solution
   

$$
   y \times IF = \int \Big( Q \times IF \Big) dx \quad + c
   

$$
   

## Bernoulli’s DE

$$
y' + P y = Q y^n

$$

### Solution

1. Divide both sides by $y^n$
   

$$
   y^{-n} y' + P y^{1-n} = Q
   

$$
   
1. Take $z = y^{1-n}$
   

$$
   \begin{align}
   z'
   &= (1-n) y^{(1-n)-1} y' \\   
   
   y^{-n} y' &=
   \left( \frac{1}{1-n} \right) z'
   \end{align}
   

$$
   
1. Convert into a Linear DE
   

$$
   \begin{align}
   \left( \frac{1}{1-n} \right) z' + P z
   &= Q \\   
   z' + \underbrace{(1-n) P}_{P\text{ of linear DE}} \ z
   &= (1-n) Q
   \end{align}
   

$$
   
1. Solving using Linear DE method in terms of $z$
   

$$
   \begin{align}
   \text{IF} &= (1-n) \int P dx \\   z \times \text{IF} &= (1-n) \int (Q \times \text{IF}) dx \quad + c
   \end{align}
   

$$
   
1. Put $z = y^{1-n}$ back into this
