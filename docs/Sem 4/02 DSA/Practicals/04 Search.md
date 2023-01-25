## Question

Write a C/C++/JAVA program to perform the following:
1. Initialize 10000 unique, positive and consecutive integers whose values are in the range $[0-9999]$, and store them in serial/ascending order in an array `A[10000]`. i.e. you are already initializing the array in the sorted order using an iterative statement.
2. Implement the Linear Search and Binary Search algorithms and DISPLAY the `<position in the array, search_time>` for the following test cases: $5000, 9997, 50000$

(assume index of the first element in array is 0)

## Algorithm

### Pseudocode

```pseudocode
Algorithm linearSearch()
	OUTPUT position of data
	
	pos <- -1
	for i<-0 to (n-1) do
		if a[i] = data
			pos = i
	return pos

Algorith binarySearch()
	OUTPUT position of searched element
	
	pos <- -1
	f <- 0
	l <- n-1

	while f <= l
		if a[m] < data
			f = m+1
		else if a[m] > data
    	l = m-1
    else
    	pos = i
  return pos
```

### Time Complexity

|  Algorithm   |  Complexity   |
| :----------: | :-----------: |
| linearSearch |    $O(n)$     |
| binarySearch | $O(\log_2 n)$ |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U
package Programs;
class p04
{
	static int n = 10000;
	static int n1 = 5000,
		n2 = 9997,
		n3 = 50000;
	static int[] a = new int[n];
	
	static float linearSearchTime, binarySearchTime;

	public static void initialize()
	{
		for(int i = 0; i<n; i++)
		a[i] = i;
	}

	public static int linearSearch(int d)
	{
		int pos = -1;
		long startTime = System.nanoTime();

		for(int i = 0; i<n; i++)
			if(a[i] == d)
			{
				pos = i;
				break;
			}
		
		long endTime = System.nanoTime();
		linearSearchTime = (endTime - startTime)/1000f;
		return pos;
	}
	
	public static int binarySearch(int d)
	{
		int pos = -1;
		long startTime = System.nanoTime();

		int f = 0, l = n-1, m;
		while(f <= l)
		{
			m = (f+l)/2;
			
			if(a[m] < d)
			f = m + 1;
			else if ( a[m] > d)
			l = m - 1;
			else // a[m] == d
			{
				pos = m;
				break;
			}
		}
		
		long endTime = System.nanoTime();
		binarySearchTime = (endTime - startTime)/1000f;

		return pos;
	}
	
	public static void linearSearchDisplay()
	{
		System.out.println( 
		"Linear Search \n" +
		"Input \t Index \t Search Time(microsec) \n" +
		n1 + "\t" + linearSearch(n1) + "\t" + linearSearchTime + "\n" +
		n2 + "\t" + linearSearch(n2) + "\t" + linearSearchTime + "\n" +
		n3 + "\t" + linearSearch(n3) + "\t" + linearSearchTime + "\n"
		);
	}

	public static void binarySearchDisplay()
	{
		System.out.println(
		"Binary Search \n" +
		"Input \t Index \t Search Time(microsec) \n" +
		n1 + "\t" + binarySearch(n1) + "\t" + binarySearchTime + "\n" +
		n2 + "\t" + binarySearch(n2) + "\t" + binarySearchTime + "\n" +
		n3 + "\t" + binarySearch(n3) + "\t" + binarySearchTime + "\n"
		);
	}
	
	public static void main(String[] args)
	{
		initialize();
		linearSearchDisplay();
		binarySearchDisplay();
	}
}
```

## Test Cases

```
Linear Search 
Input   Index   Search Time(microsec) 
5000    5000    108.2
9997    9997    221.8
50000   -1      220.7

Binary Search 
Input   Index   Search Time(microsec) 
5000    5000    2.1
9997    9997    1.1
50000   -1      0.7
```