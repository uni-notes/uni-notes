represent periodic signals in terms of cosines and sines.

## Periods

A period signal repeats its pattern at some period $T$.

Fourier series of period signal can be used to analyze the signal in another domain.

If $f(t)$ is a function with period T, then

$$
f(t+nT) = f(t) \quad \forall n

$$

|             Function             |      Period       |
| :------------------------------: | :---------------: |
|    $\cos \theta, \sin \theta$    |      $2 \pi$      |
| $\cos (n\theta), \sin (n\theta)$ | $\dfrac{2\pi}{n}$ |

## Fourier Series

of a function $f(x)$ of period $2\pi$ in the interval $[-\pi, +\pi]$, is defined as

$$
f(x) =
\frac{a_0}{ \textcolor{orange}{2} } +
\sum\limits_{n=1}^\infty a_n \cos(nx) +
\sum\limits_{n=1}^{\infty} b_n \sin (nx)
$$

It is always continuous.

Whenever possible, we have to make it into regular summation, ie from $1 \to \infty$.

### Fourier Constants

$$
\begin{align}
a_0 &= \frac{1}{\pi} \int\limits_{-\pi}^{\pi} f(x) \cdot dx \\
a_n &= \frac{1}{\pi} \int\limits_{-\pi}^{\pi} f(x) \textcolor{orange}{\cos(nx)} \cdot dx \\
b_n &= \frac{1}{\pi} \int\limits_{-\pi}^{\pi} f(x) \textcolor{orange}{\sin(nx)} \cdot dx

\end{align}
$$

## Sum of Functions

$$
FS(g_1 \pm g_2) = FS(g_1) \pm FS(g_2)
$$

## Dirchelet Condition

Even though the function that the FS represents may be discontinuous, the FS itself will be continuous

$$
\text{FS}
\stackrel{\text{converges}}{\longrightarrow}

\begin{cases}
f(a) &, a = \text{Continuous Point} \\\dfrac{f(a^-) + f(a^+)}{2} &, a = \text{Discontinuous Point}
\end{cases}
$$

## Even/Odd Functions

|                |                           Even                           |                         Odd                         |
| :------------: | :------------------------------------------------------: | :-------------------------------------------------: |
|    $f(-x)$     |                          $f(x)$                          |                       $-f(x)$                       |
| Fourier Series | $\dfrac{a_0}{2} + \sum\limits_{n=1}^\infty a_n \cos(nx)$ |       $\sum\limits_{n=1}^\infty b_n \sin(nx)$       |
|     $a_0$      |        $\dfrac{2}{\pi} \int\limits_0^\pi f(x) dx$        |                          0                          |
|     $a_n$      |   $\dfrac{2}{\pi} \int\limits_0^\pi f(x) \cos(nx) dx$    |                          0                          |
|     $b_n$      |                            0                             | $\dfrac{2}{\pi} \int\limits_0^\pi f(x) \sin(nx) dx$ |

This is because $\int f(x) dx = 0$ when $f(x)$ is even

### Note

Consider

$$
\begin{align}
f(x) &= \begin{cases}
	g_1(x), & (-a, 0) \\	g_2(x), & (0, a)
\end{cases}\\\implies 
f(x) &= \begin{cases}
\text{Even}, & g_1(-x) = +g_2(x) \\\text{Odd},  & g_1(-x) = -g_2(x)
\end{cases}
\end{align}
$$

### Grahphically

We can also plot the points for $x = \{-\pi, 0, - \pi\}$. Connect the points.

| Function Type | Symmetric about |
| ------------- | --------------- |
| Even          | Y axis          |
| Odd           | Origin          |

## Sine/Cosine Series

Special types of series, where we represent the fourier series in terms of $\sin$ alone or $\cos$ alone in half interval ==$(0,\pi)$==

Sine/Cosine series may be asked for an **odd/even** function.

|                |                      Cosine Series                       |                     Sine Series                     |
| :------------: | :------------------------------------------------------: | :-------------------------------------------------: |
| Fourier Series | $\dfrac{a_0}{2} + \sum\limits_{n=1}^\infty a_n \cos(nx)$ |       $\sum\limits_{n=1}^\infty b_n \sin(nx)$       |
|     $a_0$      |        $\dfrac{2}{\pi} \int\limits_0^\pi f(x) dx$        |                          0                          |
|     $a_n$      |   $\dfrac{2}{\pi} \int\limits_0^\pi f(x) \cos(nx) dx$    |                          0                          |
|     $b_n$      |                            0                             | $\dfrac{2}{\pi} \int\limits_0^\pi f(x) \sin(nx) dx$ |

## Arbitrary Interval

Fourier series of $f(x)$ of period $2l$ defined in the interval $(-l, l), l \in R$ is

$$
f(x) =
\frac{
	a_0
}{
	\textcolor{orange}{2}
}
+ \sum_{n=1}^\infty a_n \cos \left(
	\frac{n \textcolor{hotpink}{\pi} x}{\textcolor{hotpink}{l}}
\right)
+ \sum_{n=1}^\infty b_n \sin \left(
	\frac{n \textcolor{hotpink}{\pi} x}{\textcolor{hotpink}{l}}
\right)
$$

$$
\begin{align}
a_0 &= \frac{1}{\textcolor{hotpink}{l}} \int\limits_{-l}^l f(x) dx \\a_n &= \frac{1}{\textcolor{hotpink}{l}} \int\limits_{-l}^l f(x) \cos \left(
	\frac{n \textcolor{hotpink}{\pi} x}{\textcolor{hotpink}{l}}
\right) dx \\b_n &= \frac{1}{\textcolor{hotpink}{l}} \int\limits_{-l}^l f(x) \sin \left(\frac{n \textcolor{hotpink}{\pi} x}{\textcolor{hotpink}{l}} \right) dx
\end{align}
$$

### Changes in Interval

|              From               |                   To                    |    For     |
| :-----------------------------: | :-------------------------------------: | :--------: |
|          $(-\pi, \pi)$          |                $(-l, l)$                |     FS     |
|           $(0, \pi)$            |                $(0, l)$                 | CS and SS  |
|           $\cos(nx)$            | $\cos \left( \frac{n \pi x}{l} \right)$ | FS, CS, SS |
|           $\sin(nx)$            | $\sin \left( \frac{n \pi x}{l} \right)$ | FS, CS, SS |
| $\frac{1}{\pi} \int_{-\pi}^\pi$ |        $\frac{1}{l} \int_{-l}^l$        | FS, CS, SS |
|   $\frac{2}{\pi} \int_0^\pi$    |         $\frac{2}{l} \int_0^l$          | FS, CS, SS |

## Bernoulli’s Integration Chain Rule

$$
\int(uv) dx =
u v_1 -
u' v_2 -
u'' v_3 -
\ldots -
u^{(n-1)} v_n

$$

|          Term          |     Meaning     |
| :--------------------: | :-------------: |
|         $u, v$         | Given Functions |
|    $u', u'', \dots$    |   Derivatives   |
| $v_1, v_2, v_3, \dots$ |    Integrals    |

