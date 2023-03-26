Based on complete binary trees

## Types

| Type     | Property            |
| -------- | ------------------- |
| Min-Heap | $v \le$ descendants |
| Max-Heap | $v \ge$ descendants |

## Implementation

### Array

1. Root = 0
2. Left Child = $2i+1$

Consider a node with index $i$

| Node                | Value     |
| ------------------- | --------- |
| Root of entire tree | 0         |
| Left Child          | $2i+1$    |
| Right Child         | $2i+2$    |
| Parent              | $(i-1)/2$ |

### Linked List

$$
\fbox{l}
\fbox{data}
\fbox{r}
\fbox{parent}
\notag
$$

## Insertion

1. Find insertion point
2. Store there
3. Verify heap property
4. if not satisfied, [up bubbling](#heapification)

## Deletion

1. Remove the root element (We cannot remove a particular element)
2. Replace node with the last node of the subtree
3. Verify Heap property
4. if not satisfied, [down bubbling](#heapification)

## Heapification

up/down heap bubbling

1. compare 2 elements
2. swap if condition is not satisfied

```java
void maxHeapify(int[] a, int n, int i)
{
  int largest = i, // assuming root is the largest
  		l = (2*i) + 1,
  		r = (2*i) + 2;
  
  if(l<n && a[l]>a[largest])
    largest = l;

  if(r<n && a[r]>a[largest])
    largest = r;
  
  if(largest != i)
    // swap root with largest node
    swap(a[i], a[largest])
}
```

## Applications

1. [Heap Sort](07_Sorting.md#Heap Sort)
2. Order Statistics
3. [Priority Queue](#Priority Queue)

## Priority Queue

- Max-Heap for max-priority queue
- Min-Heap for min-priority queue
