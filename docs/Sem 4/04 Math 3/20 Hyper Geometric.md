if $a$ and/or $b$ are negative integers, then it will become a polynomial of degree $n$.

## General Form

something

$$
(1-x^2)
$$

## Standard Form

$$
x(1-x) y'' +
\Big[c - (a+b+1)x \Big]y' -
(ab)y = 0
\label{standard}
$$

where $a, b, c$ are real constants

$x=0, x=1$ are the regular singular points of $\eqref{standard}$

By Frobenius Series method, at regular singular points, we get 2 initial roots

- $m=0$
- $m=1-c$

|              |                            $m=0$                             |                           $m=1-c$                            |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Solution $y$ | $1 + \frac{a \cdot b}{1 \cdot c}x + \frac{a(a\textcolor{hotpink}{+1}) \cdot b(b\textcolor{hotpink}{+1})}{1(1\textcolor{hotpink}{+1}) \cdot c(c\textcolor{hotpink}{+1})} x^2 + \\
\frac{a(a\textcolor{hotpink}{+1})(a\textcolor{orange}{+2}) \cdot b(b\textcolor{hotpink}{+1})(b\textcolor{orange}{+2})}{1(1\textcolor{hotpink}{+1})(1\textcolor{orange}{+2}) \cdot c(c\textcolor{hotpink}{+1})(c\textcolor{orange}{+2})} x^3 + \dots$ | $x^\textcolor{hotpink}{1-c} \times F(a+\textcolor{hotpink}{1-c}, b+ \textcolor{hotpink}{1-c}, 2-c, x)$ |
|              |      $F(a, b, c , x) = F(b, a, c, x)$<br />Commutative       |                                                              |

|        Constant        |                 Outcome                  |
| :--------------------: | :--------------------------------------: |
| $a \le 0$ or $b \le 0$ | Series breaks off into finite polynomial |
|        $c\le 0$        |          Solution doesn’t exist          |
