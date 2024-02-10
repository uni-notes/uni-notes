# Latex

## Enclosed text
```latex
\enclose{circle}{\enclose{circle}{S_5}} % mathjax

\textcircled{R} % latex/katex
```
## Subset
```latex
abc
\overset{
  \substack{a=1\\b=2\\c=3}
}{
  =
}
c
```
## Mathclap
Basically equivalent to `position: absolute` in CSS
Its position does not affect other elements

```latex
\mathclap{\text{Long Text}}

% always user for underbrace and overbrace
x \underbrace{y}_{\text{Long Text}} z \\
x \underbrace{y}_{\mathclap{\text{Long Text}}} z
```