## Question

IMPLEMENTATION of MERGE-SORT (use Recursive Version) using key field
(descending order)

1. Initialize 10000 positive, integer, Random Numbers whose values are in the range [0-9999] and store them in in an input array A[10000]. i.e. you are initializing the array with random numbers using an iterative statement. (you can use built-in random number generator function. assume that duplicate values are permitted).
2. Implement the Merge-Sort Algorithm (recursive version) to sort in descending order. Measure the time to do sorting: use built-in timer function.
3. Store the output in a text file: `mergeout.txt`
4. Display the first 7 records and last 7 records of the output file. (use unix commands `head -7 mergeout.txt` `tail -7 mergeout.txt`)

## Algorithm

### Pseudocode

```pseudocode
Algorithm mergeSort(A, p, r)
	if p < r
    q ← floor( (p + r)/2 )
    
    mergeSort (A, p, q)
    mergeSort (A, q+1, r)
    
    mergeAsc(A, p, q, r) // or mergeDesc(A, p, q, r)
    
Algorithm mergeDesc(A, p, q, r)
  n1 ← q-p+1
  n2 ← r-q

  Let L[0…(n1-1)]
  Let R[0…(n2-1)]

  for i ← 0 to n1-1
	  L[i] ← A[p+i]
  for i ← 0 to n2-1
	  R[i] ← A[q+1+i]

  i ← 0
  j ← 0
  k ← p

  while i<n1 and j<n2
	  if L[i] >= R[j]
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

### Time Complexity

| Algorithm |   Complexity    |
| :-------: | :-------------: |
| mergeSort | $O(n \log_2 n)$ |

## Source Code

```java
// Ahmed Thahir 2020A7PS0198U
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

class p05
{
	static int n = 10000;
	static int[] a = new int[n];
	static String outputFile = "h:\\
My Drive\\
Notes\\
Sem 4\\
02 DSA\\
Practicals\\
"
      + "mergeout.txt";
	
	static float sortTime;
	
	public static void initialize()
	{
		for(int i = 0; i<n; i++)
			a[i] = (int) ( Math.random() * n );
	}
	
	public static void mergeSort(int[] a, int p, int r)
	{
		if(p<r)
		{
			int q = (p+r)/2;
			// automatically floor, cuz java doesn't typecast into decimal
			
			mergeSort(a, p, q);
			mergeSort(a, q+1, r);
			
			mergeDesc(a, p, q, r);
		}
	}
	
	public static void mergeDesc(int[] a, int p, int q, int r)
	{
		int n1 = (q-p) + 1,
		n2 = r-q;
		
		int[] L = new int[n1],
		R = new int[n2];
		
		for (int i = 0; i < n1; i++)
		L[i] = a[p+i];
		for (int j = 0; j < n2; j++)
		R[j] = a[q+1+j];
		
		int i = 0,
		j = 0,
		k = p;
		
		while (i<n1 && j<n2)
		{
			if(L[i] >= R[j])
			// will be <= for ascending
			{
				a[k] = L[i];
				i++;
			}
			else
			{
				a[k] = R[j];
				j++;
			}
			k++;
		}
		while(i<n1)
		{
			a[k] = L[i];
			i++;
			k++;
		}
		while(j<n2)
		{
			a[k] = R[j];
			j++;
			k++;
		}
	}
	
	public static void display()
	{
		int z = 7;
		
		System.out.println("Head");
		for(int i = 0; i<z; i++)
			System.out.println(a[i]);
		
		System.out.println("\nTail");
		for(int i = n-z; i<n; i++)
			System.out.println(a[i]);
	}
	public static void write() throws FileNotFoundException
	{
		PrintWriter writeToMyFile = new PrintWriter( new File(outputFile) );
		for(int i = 0; i<n; i++)
			writeToMyFile.format("%s\n", a[i]);
	}
	public static void main(String[] args) throws FileNotFoundException
	{		
		initialize();
		
		long startTime = System.nanoTime();

		mergeSort(a, 0, n-1);

		long endTime = System.nanoTime();
		sortTime = (endTime - startTime)/1000f;
		System.out.println("Sort Time: " + sortTime + " microsec\n");

		display();
		write();
	}
}
```

## Test Cases

```
Sort Time: 3771.6 microsec

Head
9998
9998
9997
9996
9996
9993
9987

Tail
6
4
4
3
1
0
0
```