# Tensors

(Not for exam)

Tensors are $n$-dimensional arrays, which keep track of the gradient of each element in the array.

They are optimized for parallel computing and GPU-utilization, but more memory-intensive than regular arrays.

In lazy mode, the operations are not executed until required

Each tensor has

- ID
- List of inputs
- operation performed
- cached_data_output

Tracking gradients is expensive, so

```python
x = ndl.Tensor(
	[1],
  dtype = "float32"
)

sum = 0
for i in range(100):
  sum += (x**2).detach()
```

Not using `detach()` will result in tracking the inputs and operations performed unnecessarily

## Broadcasting

Efficient, as it does not copy any data

Rather than repeating the same value multiple times for matrix multiplication