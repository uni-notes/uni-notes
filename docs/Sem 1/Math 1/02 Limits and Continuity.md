## Limits

Let $f$ be defined @ all points in some neighborhood of a point $x_0$

Then $L = \lim\limits_{x \to x_0} f(x)$ is limit for $f(x)$ when $x \to x_0$ if
for a given $\epsilon > 0$, there exists a $\delta > 0$ such that $|x-x_0| < \delta \implies |f(x)-L| < \epsilon$

## Finding $\delta$

1. Solve the inequality $f(x) - L < \epsilon$ for $x$
2. Find an interval $(a, b)$ such that $a \le x_0 \le b$
3. Choose $\delta = \min (x_0-a, b - x_0)$

This choice places the interval $(x_0 - \delta, x_0 + \delta)$ within $(a, b)$

## One-sided Limits

Let $f$ be defined at all points in the neigborhood of $x_0$ (in particular to right of $x_0$), then $f$ is said to have the right-hand limit $L$, when $x$ approaches $x_0$ from the right if the following conditions are satisfied:

For a given $\epsilon > 0$, there exists a $\delta > 0$ such that

- $x_0 < x < x_0 + \delta$
- $|f(x) - L| < \epsilon$

The limit is represented as

$$
L = \lim_{x \to {x_0}^+} f(x) = f({x_0}^+)
$$

Similarly, we define the left-hand limit

While working on one-sided problms, we proceed as follows

$$
\begin{align}
f({x_0}^+) &= \lim_{h \to 0} f(x_0 + h), & h > 0 \\f({x_0}^-) &= \lim_{h \to 0} f(x_0 - h), & h > 0
\end{align}
$$

## Continuity

A function $f(x)$ is continuous @ a point $x_0$ if the following conditions are satisfied

1. $f(x_0)$ exists
2. $\lim_{x \to x_0} f(x)$ (Both LHL and RHL) exists
3. $\lim_{x \to x_0} f(x) = f(x_0)$

### Note

If $f$ and $g$ are continuous functions in a domaind D, then the following functions are also continuous in all points of F

$$
f \pm g \\fg \\
\frac f g \\kf, &\text{k =  const}
$$

The following functions are known to be continuous in their domain of definition

1. polynomial
2. exponential
3. trignometric
