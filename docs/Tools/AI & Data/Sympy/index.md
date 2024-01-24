# Sympy

## Import

```python
import sympy as smp
```

## Basics

Declaring symbols

```python
x = smp.symbols("x")
x = smp.symbols("x", real=True, positive=True)

x, y, z = smp.symbols("x y z")
```

Declaring functions

```python
f = smp.symbols("f", cls=smp.Function)
```

Numbers

```python
x = smp.Rational(5, 1)
frac = smp.Rational(1, 2)
```

Useful Functions

```python
y = smp.sin(x)
z = x**2 + y**2
```

```python
z.factor()
z.expand()
z.simplify()
```

## Solve

```python
smp.solve(z, x) ## find value of x that makes z(x) = 0 
smp.solve(z, y)
```

## Convert to Numerical

Lambdify

```python
expr = smp.sin(x) + smp.sin(y)
expr_f = smp.lambdify([x, y], expr)
```

Substitute

```python
expr = smp.sin(x) + smp.sin(y)
expr.subs([
	(x, 10)
])
expr.subs([
  (x, 10),
  (y, 5)
])
expr.subs([
  (x, 10),
  (y, smp.sin(x))
])
```

## Calculus

### Differentiation

```python
dfdx = smp.diff(f) ## f = function symbol, which is a function of  
dfdx_sub = dfdx.sub([
	(g, smp.sin(x))
])

dfdx_sub_value = dfdx_sub.doit()
```

### Integration

Indefinite

```python
## does not give +c
smp.integrate(
  expr,
  x
)
```

Definite

```python
smp.integrate(
  expr,
  (x, 0, smp.log(4))
)
```

## Vectors

```python
u1, u2, u3 = smp.symbols("u1 u2 u3")
u = smp.Matrix([u1, u2, u3])

v1, v2, v3 = smp.symbols("v1 v2 v3")
v = smp.Matrix([v1, v2, v3])
```

```python
2*u + v
u.norm()

u.dot(v)
u.cross(v)
```