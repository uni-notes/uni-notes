## Basics

- 0/1 (bit)
- 8 bits = 1 byte
- 1024 bytes = 1KB

`print("hello")` $\to$ 0/1

Language translators

- Compilers ()
  C, C++, Pascal

- Interpretors ()
  Python, Javascript, 

- Python is dynamic

    ```python
    x = "hi"
    x = 304
    ```

- Static (compilation)

    ```c
    #include <stdio.h>
    // standard input output
    // header file
    // libraries
    
    int main()
    {
      // code
      return 0;
    }
    
    def my_func():
      print("hi")
    ```

## Variables

```c
int x; // declaration
x=5; // initialization

int x = 5; // combined
```

## Tokens

- atomic unit of a code

- Key words

    ```c
    return
      void
    ```

- Identifiers

    ```
    int x = 3;
    ```

## Code Block

group of code within `{ ... }`

## Structure

```python
print("hi")
print("hello")
```

```c
#include <stdio.h>

void print_on_screen()
{
  printf("hi"); // ; = terminator
}

int main() // driver function of your program
{
  // code
  print_on_screen();

  int x = 4; // 
  
  return 0;
}
```

## Data Types

### Primitive

|           |                                                   |      |      |
| --------- | ------------------------------------------------- | ---- | ---- |
| `void`    | (nothing)                                         | 0    |      |
| `boolean` | True/False<br />true/false                        | 1    |      |
| `char`    | ‘H’                                               | 1    | %c   |
| `int`     | 34444                                             | 2    | %d   |
| `float`   | 334545.345534                                     | 4    | %f   |
| `double`  | 334545345534334545345534.334545345534334545345534 | 8    | %    |

### User-defined

(not in scope of current exam)

## Math Operators

$$
+
-
*
/
\%
$$

- `int`/`int` = `int`

- B
- ^
- Multiplication/Division (whichever comes first)
- Addition/Subtraction (whichever comes first)

### Type Casting

change in data type of a variable for a momentary purpose

implicit (automatic)

int/int $\to$ int

- 3/2 = 1.5
- 3/2 = 1

int/float $\to$ float

- larger data type

```c
int x=5;
float z = 5/(double) 5;
```

explicit(user-defined) type casting

```c
int x = -5;
int y = 2;

-5/2; // -2.5 -2
5/-2; // -2.5 -2

5/-2.0f // -2.5
```

### Relational Operators

$$
> ,
\ge ,
< , \le,
==,
\ne,
!=
$$

- $=$ and $==$
    - assignment
    - equality operator

```c
const float pi = 3.14;
```

## Logical Operators

$$
! \\
\&\& \\
||
$$

$$
[!0 \&\& 1 || 3\&\&!1]
$$

```c
int main()
{
  int x;
  
  // take input from user
  scanf("%d", &x); // storing the value in the address of x
  
  printf("%d", x);
  
  
  return 0;
}

// x = 5
```

```c
int main()
{
  int x = 5;
  
  printf("%d", ++x); // 6
  printf("%d", x); // 6
  
  return 0;
}
```