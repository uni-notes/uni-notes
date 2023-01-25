## UNIX Process Creation

If $P_1$ ***spawns*** $P_2$

- $P_1$ is parent
- $P_2$ is child

## `fork()`

function using which new processes become child processes of the caller

- No parameters
- returns 0 to child process
- returns process ID of the child to the parent

Both parent and child will immediately execute after the `fork()`

UNIX makes an exact copy of the parent’s Stack, Heap, Data, and Code in another sequence of memory locations.

==Any change of variables by parent process won’t affect the child process’ values, and vice-versa==

If there are $n$ `fork()` one after each other,

```c
void main()
{
  fork(); // 1
  fork(); // 2
  ...;
  fork(); // n
}
```

- Total number of processes $= 2^n$
- Total number of child processes $= 2^n - 1$

## Code 1

The output after `fork()`

```c
#include <unistd.h>
#include <stdio.h>

void main()
{
  int pid;
  pid = fork();
  
  print("Hello\n");
}
```

```
Hello
Hello
```

## Concept 2

The order of execution **may be**

- Parent then child
- or, Child then parent

This depends on the system

Value of `pid`

- Child (=0)
- Parent (>0)
- Unsuccessful (<0)

```c
#include <unistd.h>
#include <stdio.h>

void main()
{
  int pid;
  pid = fork();
  
  if(pid == 0)
    printf("Child\n");
  else if(pid > 0)
    printf("Parent\n");
  else
    printf("Unsuccessful Fork");
}
```

## Concept 3

Let’s say both the parent and child get the same variable, then

- Changes in parent process will only affect the value in the parent process
- Changes in child process will only affect the value in the child process

This is because each process has its own address space; any modifications will be independent of each other

```c
#include <unistd.h>
#include <stdio.h>

void main()
{
  int a = 0;
  
  int pid;
  pid = fork();
  
  if(pid == 0)
  {
    a = a+10;
    printf("Child %d\n", a);
  } else if(pid > 0)
  {
    a = a+5;
    printf("Parent\n");
  } else
    printf("Unsuccessful Fork");
}
```

```
Parent 5
Child 10
(or)
Child 10
Parent 5
```