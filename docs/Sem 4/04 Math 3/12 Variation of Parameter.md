## Variation of Parameter

for finding particular solution $y_p$

more suitable if the RHS function is a ==$\log, \tan, \cot, \sec, \csc,$ hyperbolic==

1. Find general solution
   

$$
   y_g = c_1 \textcolor{orange}{y_1} + c_2 \textcolor{orange}{y_2}
   

$$

2. Let
   

$$
   \begin{align}
   y_p &= v_1 y_1(x) + v_2 y_2(x), \text{where} \\   
   v_1 &= \int
   \frac{
   	\textcolor{orange}{-y_2} \cdot R(x)
   }{
   	W(y_1, y_2)
   } dx \\   
   v_2 &= \int
   \frac{
   	\textcolor{orange}{y_1} \cdot R(x)
   }{
   	W(y_1, y_2)
   } dx
   \end{align}
   

$$
   where $W(y_1, y_2)$ is the Wronskian
   
3. Complete solution $y = y_g + y_p$

