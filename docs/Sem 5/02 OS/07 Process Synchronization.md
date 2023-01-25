## Types of Processes

|                                      | Independent process | Co-operating process |
| ------------------------------------ | :-----------------: | :------------------: |
| Affect other processes               |          ❌          |          ✅           |
| Affected by other processes          |          ❌          |          ✅           |
| Share code/data with other processes |          ❌          |          ✅           |

## Process Synchronization

is used for orderly execution of cooperating processes that share a logical address space, in concurrent or parallel execution

## Types of Execution
| Execution Type | Meaning                                                      |
| -------------- | ------------------------------------------------------------ |
| Sequential     | One after another                                            |
| Asynchronous   | (Not in the course)                                          |
| Concurrent     | Multiple tasks start, run, and complete in overlapping time periods, in no specific order |
| Parallel       | Multiple tasks or subtasks of the same task that literally run at the same time on a hardware with multiple computing resources like multi-core processor |

## Producer-Consumer / Bounded-Buffer Problem
| Parts            | Role                                                         |
| ---------------- | ------------------------------------------------------------ |
| Producer Process | Produce an item                                              |
| Consumer Process | Consume an item                                              |
| Bounded-Buffer   | Contains produced items (values, records)<br />Implemented as Circular Array of fixed size |

### Bounded-Buffer Variables

| Variable | Stores                                          | Initial Value |
| -------- | ----------------------------------------------- | ------------- |
| in       | location of next item to be written by producer | 0             |
| out      | location of next item to be read by consumer    | 0             |
| counter  | number of elements in buffer                    | 0             |

### `counter` Updation

| Occurance   | Counter value |
| ----------- | ------------- |
| Production  | counter++     |
| Consumption | counter--     |

### `counter` Cases

| `counter` Value | __ is Blocked | because buffer is                                    | Problem                       |
| --------------- | ------------- | ---------------------------------------------------- | ----------------------------- |
| $= 0$           | Consumer      | empty                                                | [Busy Waiting](#Busy Waiting) |
| Buffer_Size     | Producer      | full<br />(buffer content has not yet been consumed) |                               |

## Interlocked/Interleaved Schedule

## Busy Waiting

consumes CPU cycles but no work is done

## Race Condition

Situtation when several processes access and manipulate shared data concurrently.

Final value of shared data depends on the final write operation

To prevent race condition, concurrent processes must be synchronized

## Critical-Section problem

$n$ cooperating processes all competing to use some shared data

### Critical-Section

Section where shared data is accessed/modified.

For producer-consumer problem, `counter++` and `counter--` are the critical section

### Requirement

Avoid data inconsistency

When one process is ints critical section, another process should not be alowed to enter its critical section

### Task

Design protocol/algorithms which cooperating processes can use to cooperate, ensuring that the [requirement](#Requirement) is satisfied

### Structure

Assume a process $P$ is executing indefinitely

```c
do
{
  entry_section
  	critical_section
  exit_section
    remainder_section
} while (1);
```

| Section   | Role                                                         |
| --------- | ------------------------------------------------------------ |
| Entry     | Gain access to criticial section/resource<br />Ensures that only a selected number of processes are in its critical section |
| Critical  | Code that affects common memory location                     |
| Exit      | Relinquish access to criticial section/resource<br />Allows other process to enter their critical section |
| Remainder |                                                              |

### Solution

| Solution         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| Mutual Exclusion | If process $P_i$ is executing in its critical section, no other processes can be executing in their critical sections |
| Progress         | If no process is executing in its critical section, and $\exist$ some processes that wish to enter their critical section, then the selection of the processes that will enter the critical section next cannot be postponed indefinitely |
| Bounded waiting  | Bound must exist on the number of times that other processes are allowed to enter their critical sections after a process has made a request to enter its critical section before that request is granted |

## Semaphores

Synchronization Tool/Construct

### Purpose

- Solve critical section problem
- Guard access to shared resource

### Parts

- Value

- Queue of process that are waiting on it

- 2 operations/methods associated with it
  atomic/indivisible - These operations cannot be interrupted

| Operation               | `wait(s)`                                                    | `signal(s)`                                                  |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Alternate Name          | P(s)                                                         | V(s)                                                         |
| Purpose                 | used by process to gain access to critical section/access to shared resource | used by process to inform that it has completed access of critical region/using the shared resource |
| Steps                   | - dec value of semaphore<br />- check safety for process to enter critical section | - inc value of semaphore<br />- check if semaphore value $\le 0$, or if any process is waiting |
| When<br />Output  = Yes | Process enters critical region                               |                                                              |
| When<br />Output  = No  | Process added to queue of process waiting for this semaphore |                                                              |

The number of process that are waiting in the semaphore queue

$$
= |\text{Semaphore Value}| \\\Big(\text{Value} \iff < 0 \Big)
$$

## Implementation of Counting Semaphore

```c
typedef struct
{
  int value;
  Queue of processes;
} Sephamore;
```

Assume 2 operations

- `block()` suspends the process that invokes it
  - Moves the process from run state $\to$ blocked/wait state
  - Control is transferred to CPU scheduler, which then schedules another process instead
- `wakeup(P)` resumes the execution of a blocked process $P$
  - Moves the process from blocked/wait state $\to$ ready state

```c
P1
{
  wait(s);
  access printer;
  signal(s);
}

P2
{
  wait(s);
  access printer;
  signal(s);
}
```

### `wait()` Operation

```c
void wait(Semaphore s)
{
  s.value--;
  
  if(s.value < 0)
  {
    // Semaphore is unavailable
    // process cannot access critical region

    add this process to s.queue;
    block();
  }
  
  // Semaphore available
  // Process gains access to critical region
}
```

#### For binary semaphore

```c
void wait(Semaphore s)
{ 
  if(s.value == 1)
  {
		// Semaphore available
		// Process gains access to critical region
    
    
    s.value = 0;
  }
  else
  {
    // Semaphore is unavailable
    // process cannot access critical region

    add this process to s.queue;
    block();
  }
}
```

### `signal()` Operation

```c
void signal(Semaphore s)
{
  s.value++;
  if(s.value <= 0)
  {
    // processes are waiting
    // pick up a process
    
    remove process from s.queue;
    wakeup(P);
  }
  
  // no processes are waiting for this semaphore
  
}
```

```c
void signal(Semaphore s)
{
  if( s.queue.isempty()  )
  {
    // no processes are waiting for this semaphore
      s.value = 1;
  }
  else
  {
    // processes are waiting
    // pick up a process
      
    remove process from s.queue;
    wakeup(P);      
  }
  
}
```

## Solution for bounded buffer problem using semaphores

Use 3 semaphores

| Semaphore                           |                                                  | Initial value | Producer produces a process | 0 means                                | 1 means                        | $n$ means    |
| ----------------------------------- | ------------------------------------------------ | ------------- | --------------------------- | -------------------------------------- | ------------------------------ | ------------ |
| Empty                               | Keep track of free buffers                       | $n$           | `wait()`                    | Buffer full                            |                                | Buffer empty |
| Full                                | Keep track of full buffers                       | 0             | `signal()`                  | Buffer empty                           |                                | Buffer full  |
| Mutex<br />**Mut**ual-**ex**clusion | Ensures exclusive access to buffer pool (binary) |               |                             | Access to buffer pool **not** possible | Access to buffer pool possible | N/A          |

This solution prevents [Busy Waiting](#Busy Waiting)

### Producer process

```c
do
{
  // produce an item in next_produced
  
  wait(empty);
  // check if buffer pool has empty buffer. if yes, then proceed else it waits.
  // avoids busy waiting
  
  wait(mutex); // gain access to buffer pool
  
  // add next_produced to buffer
    
  signal(mutex); // gives control to buffer pool
  signal(full); // increment value of full
  
} while (1);
```

## Consumer Process

```c
do
{
  // item next_consumed;
  
  wait(full);
  // check if buffer is empty. if true, process blocks
  // avoids busy waiting
  
  wait(mutex); // exclusive access buffer pool
  
  // remove item from buffer to nextconsumed
  
  signal(mutex);
  signal(empty); // increment value of empty
  
  // consume item in next_consumed
  
} while(1);
```

## Readers-Writers Problem

|                | Meaning                                          |
| -------------- | ------------------------------------------------ |
| Reader Process | Reads from shared resource                       |
| Writer Process | **Reads ==and/or== writes** from shared resource |

### Problem Statement

Multiple readers can access shared resource

Only 1 writer can have **exclusive access** to shared resource

| Process 1 | Process 2 |      |
| --------- | --------- | :--: |
| Read      |           |  ✅   |
| Read      | Read      |  ✅   |
| Read      | Write     |  ❌   |
| Write     |           |  ✅   |
| Write     | Read      |  ❌   |
| Write     | Write     |  ❌   |

### Variables

|                                                   | `readcount`                                                  | `mutex`                       | `rw_mutex`                                                   |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| data type                                         | integer                                                      | binary semaphore              | binary semaphore                                             |
| Purpose                                           | Keep track of number of readers accessing shared resource @ a time<br /><br />Give access to first reader<br />Relinquish access after last reader | Control access to `readcount` | Control access to shared resource                            |
| Used by                                           | N/A                                                          | Readers                       | 1^st^ reader - gain exclusive access `wait(rw_mutex)`<br />Last reader - relinquish access `signal(rw_mutex)`<br /><br />All writers `wait(rw_mutex)`, `signal(rw_mutex)` |
| initial value                                     | 0                                                            | 1                             | 1                                                            |
| when a reader/writer accesses the shared resource | inc                                                          | 0                             | 0                                                            |
| when a reader/writer finishes reading             | dec                                                          | 1                             | 1                                                            |

### Writer Process

```c
do
{
  // entry section
  wait(rw_mutex);	// gain exlclusive access by this writer process to shared resource
  
  // critical section
  // perform read and/or write operation

  
  // exit section
  signal(rw_mutex);	// relinquish access by this writer process to shared resource
  
} while(1);
```

### Reader Process

```c
do
{
  wait(mutex); // exclusive access to readcount

  readcount++;
  if(readcount == 1) // 1st reader
    wait(rw_mutex); // gain exclusive access by all readers to shared resource
  
	signal(mutex); // relinquish access to readcount
  
  // read from shared resource
  
  wait(mutex); // exclusive access to readcount

  readcount--; 
  if(readcount == 0) // last reader
    signal(rw_mutex); // relinquish access by all readers to shared resource
  signal(mutex); // relinquish access to readcount
  
} while(1);
```

