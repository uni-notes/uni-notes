## `getpid()`

Used to display the id of the process that invokes it

## `getppid()`

Get parent’s process ID

## `wait(NULL)`

==parent process waits for completion of **any one** of its children==

Parent initially moves to wait state, then comes to ready queue

### Dependencies

- `#include <sys/types.h>`
- `#include <sys/wait.h>`

### return value

pid of the terminated child

### exit status of child

integer value

- +ve: sucessful termination
- -ve : unsuccessful termination

```c
pid_t wait(int *status);

int status;
pid = wait(&status);
```

## Code 1

```c
#include <stdio.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/wait.h>
void main()
{
  int rv, a;
  rv = fork();
  
  if(rv == 0)
  {
		printf("Hello\n");
    printf("Child PID is %d\n", getpid());
    printf("My parent's PID is %d\n", getppid());
  }
  else if(rv > 0)
  {
    a = wait(NULL);
    printf("Parent PID is %d\n", getpid());
    printf("Parent: The child that terminated is %d\n", a);
  }
  else
  {
    printf("Unsuccessful");
  }
}
```

```
Hello
Child PID is 1676
My parent's PID is 1672
Parent PID is 1672
Parent: The child that terminated is 1676
```

