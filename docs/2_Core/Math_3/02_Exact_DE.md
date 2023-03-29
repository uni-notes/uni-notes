## Family of Curves

$$
\begin{aligned}
f(x, y) &= c \\
d( \ f(x, y) \ ) &= d(c) \\
f_x dx + f_y dy &= 0
\end{aligned}
$$

This last step

- looks like [Homogeneous Equation](01 Intro#Homogeneous Equation)
- is the **exact differential equation** of the given curve

The solution is the given equation, and from that we derived the exact differential equation.

## Solution of a DE

Consider a first-order differential equation of the form

$$
M dx + N dy = 0
$$

**if** there happens to be a function $f(x, y)$ such that

$$
\begin{aligned}
f_x = M(x, y),
\quad
f_y &= N(x, y) \\
\frac{\partial f}{\partial x} dx
+
\frac{\partial f}{\partial y} dy
&= 0 \\
d( f(x, y) ) &= 0 \\
f(x, y) &= c
\end{aligned}
$$

The final step is the general solution of the given differential equation.

## Exact DE

is a differential equation where $M_y = N_x$

## Shortcut Method for Exact DE

This is only for ==exact DE==

Consider this DE

$$
(\ y + y \cos(xy)  \ )dx + (\ x + x \cos(xy) \ ) = 0
$$

1. Check if the given DE is exact

2. Put integration sign for both sides
  
$$
\int (\ y + y \cos(xy)  \ )  dx + \int(\ x + x \cos(xy) \ )  dx = \int 0
$$
   
3. Simplifications
   1. Treat $y$ as a constant in the $dx$ integral
   2. Drop all terms containing $x$ in the $dy$ integral
      think like this: **drop your ex**
      example
      - $y \cos(x) \to 0$
      - $x \cos(x) \to 0$
      - $y + y \cos(x) \to y$
   
$$
\int (\  y + y \cos(xy) \ ) dx + \int (0 + 0) dy = c
$$
   
4. Integrate

$$ \begin{aligned} yx + y \left( \frac{ \sin xy }{ y } \right) &= c \\
yx + \sin(xy) &= c
\end{aligned} $$

## Exact DE Formulae

$$
\begin{aligned}
d(xy) &= xdy + y dx \\
d(x^2 + y^2) &= 2x dx + 2y dy \\
d \left(\frac{x^2 + y^2}{2} \right) &= x dx + y dy
\end{aligned}
$$

$$
\begin{aligned}
d\left(\frac{x}{y}\right) &= \frac{ydx - xdy}{y^2}
\quad \left(\frac{u}{v} \right)' \text{ formula}\\
&= \frac{1}{y}dx - \frac{x}{y^2} dy \\
d\left(\frac{y}{x}\right) &= \frac{xdy - ydx}{x^2} \\
&= \frac{1}{x}dy - \frac{y}{x^2} dx
\end{aligned}
$$

$$
\begin{aligned}
d\left(\log{ |\frac{x}{y}| }\right) &= \frac{1}{\frac{x}{y}} \left( \frac{y dx - x dy}{y^2} \right) \\
&= \frac{y dx - x dy}{xy} \\
\end{aligned}
$$

$$
\begin{aligned}
d\left( \log \| \frac{y}{x} \| \right) &= \frac{x dy - y dx}{xy}
\end{aligned}
$$

$$
\begin{aligned}
d \left( \tan^{-1} \frac{x}{y} \right) &=
\frac{1}{1 + \frac{x^2}{y^2} }
\left( \frac{y dx - x dy}{y^2} \right) \\
&= \frac{y dx - x dy}{x^2 + y^2} \\
d \left( \tan^{-1} \frac{y}{x} \right) &=
\frac{x dy - ydx}{x^2 + y^2}
\end{aligned}
$$

## IDK

You cannot integrate $\int f(x,y) \ dx$ wrt to $dx$ alone

it is only possible for something sir said and double integration (there $dy$ will also be there in outer integral)
