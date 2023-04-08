## Algorithms

| Algorithm |                           Working                           | In-Place | Worst                 | Avg             | Best            |
| :-------: | :---------------------------------------------------------: | :------: | :-------------------- | :-------------- | --------------- |
|  Bubble   |                elements swapped with bubble                 |    ✅     | $O(n^2)$              | $O(n^2)$        | $O(n)$          |
| Selection |             swap current element with smallest              |    ✅     | $O(n^2)$              | $O(n^2)$        | $O(n)$          |
|   Merge   |                  Recursive Divide-Conquer                   |    ❌     | $O(n \log_2 n)$       | $O(n \log_2 n)$ | $O(n \log_2 n)$ |
|   Quick   | Recursive Divide-Conquer<br /> Partition array around pivot |    ✅     | $O(n^2)$              | $O(n \log_2 n)$ | $O(n \log_2 n)$ |
| Insertion |             key compared with previous elements             |    ✅     | $O(n^2)$              | $O(n^2)$        | $O(n)$          |
|  Bucket   |             bucket of pointers to linked lists              |    ❌     | $O(n^2)$              | $O(n+k)$        | $O(n + k)$      |
|   Radix   |                         tuple-based                         |    ✅     | $O \Big( (T(n) \Big)$ |                 |                 |
|   Heap    |                          Max-heap                           |    ✅     | $O(n \log_2 n)$       |                 |                 |

## Bubble Sort

- The $j$ loop acts as a controller for no of times the inner loop runs
- The $i^{th}$ element acts as the value stored in the ‘bubble’

```pseudocode
for(int j=1; j<=n-1; j++)
	for(int i=0; i<n-1; i++)
		if(a[i+1]<a[i])
		{
			t = a[i+1];
			a[i+1] = a[i];
			a[i] = t;
		}
```

## Selection Sort

useful when memory write is a costly operation

```pseudocode
for (j=0; j<n-1; j++)
{
	m = a[j];
	pos = j;
	
	for(i=j+1; i<n; i++)
		if(a[i]<m)
		{
			m = a[i];
			pos = i;
		}
		
	a[pos] = a[j];
	a[j] = m;
}
```

## Merge Sort

1. Divide the list
2. Recursively sort the divisions
3. Merge the divisions

Let $p, r, q$ denote left, right, and middle indices

```pseudocode
Algorithm mergeSort(A, p, r)
	if p < r
    q ← floor((p + r)/2)
    
    mergeSort (A, p, q) // first half
    mergeSort (A, q+1, r) // second half
    
    mergeAsc(A, p, q, r) // or mergeDesc(A, p, q, r)
    
Algorithm mergeAsc(A, p, q, r)
  n1 ← q-p+1
  n2 ← r-q

  Let L[0…(n1-1)] // int[] l = new int[n1]
  Let R[0…(n2-1)] // int[] r = new int[n2]

  for i ← 0 to n1-1
	  L[i] ← A[p+i]
  for i ← 0 to n2-1
	  R[i] ← A[q+i+1]

  i ← 0
  j ← 0
  k ← p

  while i<n1 and j<n2
	  if L[i] <= R[j] // >= for mergeDesc
		  A[k] ← L[i]
		  i ← i+1
	  else
		  A[k] ← R[j]
		  j ← j+1
	  k ← k+1

  while i<n1
	  A[k] ← L[i]
	  i ← i+1
	  k ← k+1

  while j< n2
	  A[k] ← R[j]
	  j ← j+1
	  k ← k+1
```

## Quick Sort

After each iteration, the pivot element will move to its correct position

```pseudocode
Algorithm quickSort(a, p, r)
	if(p<r)
		q = partition(a, p, r)
		quickSort(a, p, q - 1)  // Before pivot
		quickSort(a, q + 1, r) // After pivot
		// the reason q is left out is cuz it is already placed in its correct position
    
Algorithm partition(a, p, r)
	pivot = a[r] // assuming last element as pivot
	i = p - 1
	
	for j=p to r-1
		if a[j] <= pivot
			i = i+1
			swap a[i] and a[j]
			
	swap a[i+1] with pivot
	return i+1 // this is the new position of the pivot element
```

### Randomized Quick Sort

The randomness reduces the worst-case complexity

```pseudocode
Algorithm randomizedQuickSort(a, p, r)
	if(p<r)
		q = randomizedPartition(a, p, r)
		
		randomizedQuickSort(arr, p, q - 1)  // Before pivot
    		randomizedQuickSort(arr, q + 1, r) // After pivot
    
Algorithm randomizedPartition(a, p, r)
	i = random(p, r)
	exchange a[r] with a[i]
	return partition(a, p, r)
	// same as regular quick sort partition
```

## Insertion Sort

```pseudocode
Algorithm insertionSort(a, n)
	for i = 1 to n // important
		key <- a[i]
		j <- i-1
		while j>=0 and a[j] > key
			a[j+1] <- a[j]
			j <- j-1
		a[j+1] <- key
```

## Bucket Sort

Complexity goes down, as now you’ll only be sorting subarrays.

```pseudocode
function bucketSort(array, k) is
  buckets ← new array of k empty lists
  M ← the maximum key value in the array
  
  for i = 1 to length(array) do
    insert array[i] into buckets[floor(k × array[i] / M)]
  
  for i = 1 to k do
    insertionSort(buckets[i])
  
  return the concatenation of buckets[1], ...., buckets[k]
```

### Textbook

```pseudocode
Algorithm bucketSort(a, n)
	buckets <- array of linked lists
	max <- maximum key value in the array
	
	for each entry e in S do
		k <- key of e
		remove e from s
		insert e at the end of bucket[k]
		
	for i<-0 to n-1 do
		for each entry e in bucket[i] do
			remove e from b[i]
			insert e at the end of S
```

## Radix Sort

Lexicographical sort

Tuple-based sorting for multi-dimensional element.

```pseudocode
Algorithm radixSort(s)
	INPUT sequence s of d-tuples
	OUTPUT sequence s sorted lexicographically
	
	bucketSort(S, Comparator)
```

## Heap Sort

1. Build a heap ($n$ steps)
2. Repeat ($n$ times)
   1. Remove the root node and place in the last index
   2. Rebuild max-heap

```java
void heapSort(int[] a)
{
  buildHeap(a, a.length);
  
  for(int i = n-1; i>=0; i--)
  {
    swap(a[0], a[i]);
    maxHeapify(a, i, 0)
  }
}
```

| Sort Direction | Heap Type |
| -------------- | --------- |
| Ascending      | Max-Heap  |
| Descending     | Min-Heap  |

