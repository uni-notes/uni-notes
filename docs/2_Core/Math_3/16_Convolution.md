## Definition

$$
f(t) \star g(t)
= \int\limits_0^\infty
f(t-\tau) g(\tau) \cdot d\tau
$$

$$
f(t) \star g(t) = g(t) \star f(t)
$$

## Convolution Theorem

It is used for Laplace Transform

$$
L \{ f(t) \star g(t) \}
= F(s) \cdot G(s)
$$

$$
\begin{aligned}
L^{-1} \{ F(s) \cdot G(s) \}
&= f(t) \star g(t) \\&= L^{-1}\{ F(s) \} \star L^{-1}\{ G(s) \}
\end{aligned}
$$

## Trignometric

$$
\begin{aligned}
\cos(x) &= \frac{
	e^x \textcolor{orange}{+} e^{-x}
}{2} \\
\sinh(x) &= \frac{
	e^x \textcolor{orange}{-} e^{-x}
}{2} \\
\cos(x) &= \frac{
	e^{\textcolor{hotpink}{i} x} \textcolor{orange}{+} e^{-\textcolor{hotpink}{i} x}
}{2i} \\
\sin(x) &= \frac{
	e^{\textcolor{hotpink}{i} x} \textcolor{orange}{-} e^{-\textcolor{hotpink}{i} x}
}{2i}
\end{aligned}
$$

