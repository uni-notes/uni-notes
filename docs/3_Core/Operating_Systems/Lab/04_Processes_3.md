## `execlp()`

replaces the data & text region of the calling process with the new data of program

```c
execlp(program path, exec_name, arg_1, arg_2, ... , NULL);
```

`NULL` is always the last parameter of `execlp()`

`execlp()` ==**will return to the child process only in case of error. It will go back to the parent process regardless.**== 

Hence, in the following example, “blah blah” will not execute.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

void main()
{
  int rv;
  rv = fork();
  
  if(rv == 0)
  {
    printf("I am a child process\n");
    execlp("ls", "ls", NULL);
    printf("blah blah"); 
  }
  else if(rv > 0)
  {
    wait(NULL);
    printf("Hi. I am the parent\n");
  }
  else
  {
    printf("Unsuccessful");
  }
}
```

```
I am a child process
a.out  main.c
Hi. I am the parent
```

## Code 2: Long Listing

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

void main()
{
  int rv;
  rv = fork();
  
  if(rv == 0)
  {
    printf("I am a child process\n");
    execlp("ls", "ls", "-l", NULL);
    printf("blah blah");
  }
  else if(rv > 0)
  {
    wait(NULL);
    printf("Hi. I am the parent\n");
  }
  else
  {
    printf("Unsuccessful");
  }
}

```

```
I am a child process
total 24
-rwxr-xr-x 1 runner3 runner3 16864 Oct 20 05:55 a.out
-rwxrwxrwx 1 root    root      371 Oct 20 05:55 main.c
Hi. I am the parent
```

## Code 3: Without `NULL`

==Gives **error**==: missing sentinel in functional call

This causes blah blah to be displayed

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

void main()
{
  int rv;
  rv = fork();
  
  if(rv == 0)
  {
    printf("I am a child process\n");
    execlp("ls", "ls", "-l");
    printf("blah blah");
  }
  else if(rv > 0)
  {
    wait(NULL);
    printf("Hi. I am the parent\n");
  }
  else
  {
    printf("Unsuccessful");
  }
}
```

```
main.c: In function ‘main’:
main.c:14:5: warning: missing sentinel in function call [-Wformat=]
   14 |     execlp("ls", "ls", "-l");
      |     ^~~~~~
I am a child process
blah blahHi. I am the parent
```

## Code 4: Executing another program

### `sample.c`

```c
#include <stdio.h>
void main()
{
  printf("Hi there!");
  something
}
```

```bash
cc sample.c a.out
cc sample.c -o sample.o
```

