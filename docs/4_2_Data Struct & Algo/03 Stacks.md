Data is stored one over the other

$t$ is a variable that refers to the top. Initial value is -1 (stack empty)

| Operation     | Return Type | Function                                                |
| ------------- | :---------: | ------------------------------------------------------- |
| push(element) |    void     | inserts element at top position                         |
| pop()         |   element   | removes topmost element and returns the removed element |
| top()         |   element   | returns the topmost element                             |
| size()        |     int     | returns no of elements                                  |
| isEmpty()     |   boolean   | checks if empty                                         |

```pseudocode
Algorithm push(element)
		if t = n-1
			overflow
		t = t + 1
		a[t] = element

Algorithm pop()
		if t = -1
			underflow
		t = t - 1
		return a[t]

Algorithm size()
	return (t+1)
	
Algorithm top()
	return a[t]
	
Algorithm isEmpty()
	if t = -1
		return true
```

### Applications

1. browsing history
2. undo sequence
3. chain of method calls in JVM (java virtual machine)
4. Evaluation and conversion of expressions (infix, post-fix, pre-fix)

## Infix $\to$ PostFix

| Token | Stack | Output    |
| :---: | :---- | :-------- |
|   a   |       | a         |
|   +   | +     | a         |
|   c   |       | ac        |
|     -   | -     | ac+       |

### Rules

| Input                 | Output        |
| --------------------- | ------------- |
| HIN $\leftarrow$ LCIN | HOUT, ALL OUT |
| LIN $\leftarrow$ HCIN | No change     |

### Priority

| Arithmetic | Logical |
| ---------- | ------- |
| $()$       | NOT     |
| ^          | AND     |
| $*/$       | OR      |
| $+-$       |         |

If what is coming in and what is already in have the same priority, then
the one inside is considered as the higher priority

## PostFix $\to$ Infix

Simple rules

| Token |  Stack  |          Action          |
| :---: | :-----: | :----------------------: |
|   3   |    3    |          Push 3          |
|   2   |  3, 2   |          Push 2          |
|   5   | 3, 2, 5 |          Push 5          |
|   ^   |  3, 32  |  Pop $2, 5$; Push $2^5$  |
|   +   |   35    | Pop $3, 32$; Push $3+32$ |

## Balancing of Symbols

We’re basically checking if all the brackets are matched

```pseudocode
while there are symbols in the expression do
	if symbol is variable
		do nothing
	if symbol is opening
		push it to the stack
	if symbol is closing symbol
		if stack is empty
			invalid
		else
			valid
		
if stack is empty	// once evaluation of expression is over
	valid
else
	invalid
```

| Token | Stack | Reason |
| :---: | :---: | :----: |
|       |       |        |
