Front and Rear are two variables

| Operation        | Return Type | Function                                                  |
| ---------------- | :---------: | --------------------------------------------------------- |
| enqueue(element) |    void     | inserts element at rear position                          |
| dequeue()        |   element   | removes frontmost element and returns the removed element |
| front()          |   element   | returns the frontmost element                             |
| rear()           |   element   | returns the rearmost element                              |
| size()           |     int     | returns no of elements                                    |
| isEmpty()        |   boolean   | checks if empty                                           |

## College Implementation

- $f$ shows the index of the first element
- $r$ shows the free-space available to insert the next element
  (index immediately after the last inserted element)

## Linear Queue

Textbook

```pseudocode
Algorithm enqueue(o)
	if r=N then
		return Error
	else
    Q[r]← o
    r ← r+1

Algorithm dequeue()
	if f=r then
		return Error
	else
		e ← Q[f]
		Q[f] ← null
		f ← f+1
		return e
```

## Circular Queue

```pseudocode
Algorithm size()
	return (n-f+r) mod n

Algorithm isEmpty()
	if f = r
		return true
	else
		return false

Algorithm front()
	if isEmpty() then
		return Error
	else
		return Q[f]

Algorithm dequeue()
	if isEmpty() then
		return Error

	o ← Q[f]
	Q[f] ← null
	f ← (f+1) % n

	return o

Algorithm enqueue(o)
	if size()= n-1 then
		return Error
	Q[r] ← o
	r ← (r+1) % n
```

## Questions

| Operation | $A[0]$ | $A[1]$ | $A[2]$ |  F   |  R   | Exception | Output |
| :-------: | :----: | :----: | :----: | :--: | :--: | :-------: | :----: |
|           |        |        |        |      |      |           |        |
|           |        |        |        |      |      |           |        |
|           |        |        |        |      |      |           |        |

## Applications

- buffers
- multi-threading priority
- data transfer priority
