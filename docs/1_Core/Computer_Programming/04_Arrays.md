linear data structure

primitive data structure

arrays are 0-indexed

- index start from 0

it is non-mutable (non-changeable)

statically-allocated

size of array is fixed

array is similar to tuple in python, but only same data type

collection of elements of the ==same== data type

## $\ne$ List

In python, list is a collection of elements of the same/different type

```python
list = [2, "hi", 3, 4, 5]
```

## Declaration/Creation

```c
int array[10];

char array[10];
float array[10];
double array[10];
const int length = 10;
int array[length];
```

## Initialization

```c
array[0]= 1;
array[1] = 2;

int num;
scanf("%d", &num);
array[2] = num;
scanf("%d", &num);
array[3] = num;
```

```c
for(int i=0; i<10; i++)
{
  int num;
  scanf("%d", &num);
  array[i] = num;
}

// no error
```

```c
for(int i=0; i<10; i++)
{
  scanf("%d", array[i]); // no &
}

// no error
```

```c
for(int i=1; i<=9; i++)
{
  scanf("%d", &num);
  array[i] = num;
}

// no error
```

## Display

```c
for(int i=0; i<10; i++)
{
  printf("%d\n", array[i]);
}
```

## example

```c
#include <iostream.h>

int main()
{
  int a[100]; // assuming a random number
  
  
  // input no of elements
  int n;
  printf("Please enter no of elements that you are goint to input");
  scanf("%d", &n);
  
  // input the elements
  for(int i=0; i<n; i++)
  {
    printf("Enter number at index %d: ", i);
    scanf("%d", a[i]);
  }
  
  // display marks > 60
  for(int i=0; i<n; i++)
  {
    if(a[i] > 60)
      printf("%d", a[i]);
  }
  // or
  for(int i=0; i<n; i++)
  {
    int num = a[i];
    if(num > 60)
      printf("%d", num);
  }
  
  return 0;
}
```