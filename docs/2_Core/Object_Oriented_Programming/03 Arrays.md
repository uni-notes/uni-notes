(write this somewhere else in future)

String and Object are different classes directly under `java.lang` and hence that’s why it is 

- `name.length()` - String
- `a.length` without brackets - int

## Array

a non-primitive linear data structure that is a collection of elements of the same type

starts with index 0

in java, arrays are classes

arrays are derived from `Object` class

Steps

1. declaration
2. memory allocation
   in java, memory for arrays is dynamically-allocated
3. initialization

``` java
System.out.println(inta.getClass() +
                   bytea.getClass() +
                   shorta.getClass() + 
                   
                   inta.getClass().getSuperClass() // same for bytea, shorta
                   
                   name.getClass() // name = "hello"
                   )
```

```
(Output)

class [I
class [B
class [S
class java.lang.Object
class [Ljava.lang.String //(String is an class under java.lang itself)
```

``` mermaid
graph TB
java.lang --> Object & String
Object --> arrays
```

``` java
// declaration, memory allocation, initialization in a single statement
double[] myList = {1.9, 2.4, 34, 34};  
int[] a = new int[10];

// individually -both are correct
// declaration
int[] a;
int b[];
// memory allocation
a = new int[10];
// initialization
a[3] = 100;
// accessing
int length = a.length;
length = b.length; // for 2d array, it will return no of rows
length = b[2].length; // returns the no of columns in 2nd index row

System.out.println(a[3]);

// display
for(int i = 0; i<a.length; i++)
  System.out.println(a[i]);

// sum
int sum = 0;
for(int i = 0; i<a.length; i++)
  sum += a[i];

// largest element
int max = a[0];
for(int i = 0; i<a.length; i++)
  if(a[i]>max)
    max = a[i];
```

### Array of objects

we have to dynamically

1. create the array
2. create each location

``` java
// creating the array
Student[] s = new Student[10];

// creating individual locations
for(int i = 0; i< s.length; i++)
{
  s[i] = new Student();
}

// or
for(int i = 0; i< s.length; i++)
{
  int x = inp.nextInt();
  s[i] = new Student(x);
}
```

### Passing array to method

Pass the name of the array - you're basically passing the pointer

``` java
display(arr); // without []
```

### Returning array from method

``` java
public static int[] mod()
{
  return new int[] {1,2,3};
  // or
  int[] a = new int[3]; a[0] = 1; a[1] = 2; a[2] = 3;
  return a;
  
 	// or 
  int[] a = {1,2,3};
  return a;
}
```

### Copying an Array

``` java
// same array, but new pointer
int[] b = a; 

//independent copied array
int[] b = new int[a.length];
// Thanks Firas
// Firas said just int[] b is not enough. why tho?

System.arraycopy(a, 0, b, 0, a.length); // completely copy the array
System.arraycopy(a, 1, b, 0, 2);

// skeleton of arraycopy
public static void arraycopy(
		Object src, int srcPos, 
  	Object dest, int destPos,
  	int length
	)
```

## Multidimensional Array

Array e (below code example) will look like

$$
\begin{bmatrix}
2 & 3 & 7 \\3 & 5 & 6
\end{bmatrix}
$$

``` java
// declaration
int[][] a = new int[10][20];
int[][][] b = new int[10][20][30];

int[][] e = { 
  {2, 3, 7}, // row wise allocation
  {3, 5, 6}
};

int[][] a = new int[10][10];
for(int i = 0; i < a.length; i++) //row
  for(int j = 0; j < a[i].length; j++) // column
    a[i][j] = inp.nextInt();

// or
int[][] a = new int[m][n];
for(int i = 0; i<m; i++)
  for(int j = 0; j<n; j++)
    a[i][j] = inp.nextInt();

// addition of 2 arrays
for(int i = 0; i<m; i++)
  for(int j = 0; j<n; j++)
    add[i][j] = a[i][j] + b[i][j];
```

