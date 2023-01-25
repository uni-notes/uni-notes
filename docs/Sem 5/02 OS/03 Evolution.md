## Early Computers

Something compiler

Aim was to optimize CPU time

## Batch Processing Systems

Shared computer systems

### Steps

- Operator hired
- Jobs with similar needs were batched together and run though computer, to reduce setup time

### Problems

- Stopped jobs (normal/abnormal)
- Dump (Log file)
  Status of the memory is stored into a text file
- Load device with next job
- Restart computer

### Issues

(i missed this)

CPU Burst

I/O Burst

### Solution

- Perform CPU execution and I/O concurrently
- Multi-programming and Time Sharing
    - Multi-programming $\ne$ Multi-Processing
    - Multi-Programming means using 1 CPU and performing multiple programs concurrently
    - Multi-Processing means using multiple CPUs

## Time Sharing/Multi-Tasking Systems

Extension of Multiprogramming

### Time Multiplexing

CPU switch between the programs kept in PM

This is done by dividing CPU time into Time Quanta(fixed intervals)

Involves timer = counters + clock

### Space Multiplexing

After partitioning primary memory into OS and User area, the User Area is then further partitioned for multiple users

### Advantages

- Supported user interaction
- Improved response time

### Difference from Multi-Programming

A single CPU switches between multiple users, in a way that every user feels as if they are using the CPU.

### Steps

1. P1 runs to completion
2. P1 requests I/O
3. P1 has not completed the execution, but the time quanta expired
4. Timer interrupt occurs (as the down-counting timer is over), the CPU switches to another user

### Time Delay

Time between switching between 2 users

= Count value * Time Period

### Features

- Job, CPU Scheduling
- Memory Management
- Resonable response time
    - Using Virtual memory
    - [Swap/Roll](#Swap/Roll)
- File system and disk management
- job sync and communication
- Deadlocks handling
- protection and security

### Swap/Roll

|                                                   | Swap-in/Roll-In           | Swap-out/Roll-out       |
| ------------------------------------------------- | ------------------------- | ----------------------- |
| Moving partially executed program from ___ memory | secondary $\to$ primary   | primary $\to$ secondary |
|                                                   | Basically like loading in |                         |

### Multi-Tasking

Each program of the same user is a ‘task’. The processor switches between each task.

## Computer System

|                   | Single-Processor System | Multi-Processor System |
| ----------------- | ----------------------- | ---------------------- |
| Total No of chips | 1                       | Multiple               |
| Total No of cores | 1                       | Multiple               |

Core = CPU, Register, Local Cache

## Multi-Processor System

|                             | Traditional                                                  | Modern<br />(Multi-Core)                                     |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Each processor has ___ core | 1                                                            | Multiple                                                     |
|                             | Processors connected to shared memory unit, shared I/O, shared clock, and other shared system resources | All shared resources for the cores are **within** the CPU, thereby improving performance |

- i3 has dual-core
- i5, i7 have quad-core

## Multi-Processing

Concurrent execution using multiple processors

### Increased Throughput

Execution of programs is distributed to multiple processors, hence more output

Expected increase is the number of additional processors, but this is not real. Check Amdahl's law in CA

### Improved Reliability

Even if one processor fails, complete failure is avoided. However there will be obvious delay.

### Useful for testing fault Tolerance

2 systems perform the same task and the results will be checked.

However, increased redundancy and consumption of resources.

## Desktop OS

WindowsOS, MacOS, Linux

## Mobile OS

Android, iOS

## Embedded Systems

Washing machine, dishwasher

## Realtime OS

More complex, Dynamic and have time constraints                  

eg: Industrial control systems, Weapon systems

## Distributed Systems

Multiple hardware devices are networked together.

Each device runs a subset of the ‘distributed OS’

When a process executes, the process is split into subprocesses which is sent to different nodes.

### Advantages

Some more points are there

- Data access
- Special h/w requirements
- Load balancing

## Definitions

### Throughput

No of tasks completed

### Turn around time

Time between starting of execution of a program and its completion

## Questions

1. Steps when a user double clicks MS Powerpoint icon in a Windows Desktop
     - CPU recognizes user click (Hardware Interrupt), after execution of current instruction
     - Mode changes to kernel mode
     - Identifies `.exe` file in secondary memory
     - OS loads program into primary memory
     - Allocate the CPU
     - Hardware interrupt to display on screen
2. Important activities/functions done by OS
     - [Functions of OS](#Functions of OS)
3. Suppose a computer system has 10000 bytes of memory available. Out of this, the OS occupies 5000bytes. Now it is required to run a program whose size is 7000bytes. Is this possible?
     - No, it is not possible in basic computers
     - Every program has to be loaded into primary memory for its execution
     - The memory is split into 2 parts
     - OS segment
     - Program segment (for loading the program)
     - As only 5000bytes is available for the program segment, we cannot load this program, and hence we cannot run it
     - However, we can sort this issue using virtual memory
4. What is meant by dual mode of operation in intel CPUs
     - To ensure better security and stricter access priveleges, the CPU has 2 modes of operations
     - Instructions by the user are done in user mode, with lower priveleges
     - Instructions by the user are done in kernel mode, with higher priveleges
5. Can you guess in what mode OS program is run and in what mode user program is run?
     - Kernel mode
     - User mode
6. What is the difference between hardware and software interrupt in a computer system? Can you give examples of each type.
     - Hardware interrupt are interrupts involving I/O devices. eg: keyboard input
     - Software interrupts do not include I/O devices. eg: Mathematical errors, rejected priviledged instructions, Exceptions
7. Why do you require a bootstrap loader?
     - POST Diagnostics
     - Detect and initialize devices
     - Initialize registers and primary memory
     - Load the kernel into primary memory
8. Using the Interrupt Vector Table (of an 8086 processor operating in real mode) shown below, determine the address of the ISR of a device with interrupt vector 42H. 
   
   ![image-20230101134404020](assets/image-20230101134404020.png)
   
   Answer $= \rm{4D6EA_H}$
9. Differentiate between multi-programming & multi-processing
     - Multi-programming means running multiple programs, using only 1 processor
     - Multi-processing means running one or more programs, using multiple processors

## System Call

It is the way for user instructions to perform priviledged instructions.

## Virtual Memory

Combines primary and secondary memory into a single ‘logical memory’

Load only the required part of a program into primary memory.
