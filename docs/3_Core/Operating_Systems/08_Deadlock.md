## Types of Resources

|         | Preemptable Resource                                         | Non-Preemptable Resource                                     |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|         | Can be removed from process without causing computation fail | Once allotted to a process, it cannot be removed from a process unless the process relinquishes the resource by itself |
| Example | Primary Memory can be swapped out                            | Printers                                                     |

## Necessary conditions for Deadlocks

| Condition                | Meaning                                                      |
| ------------------------ | ------------------------------------------------------------ |
| Mutual-exclusion         | Only one process can use a resource at a time                |
| Hold & Wait              | A process holding at least one resource is waiting to acquire additional resources held by other processes (hence causing a wait) |
| Non-Preemptive resources | A resource can be released only voluntarily by the process holding it, after that process has completed its task. |
| Circular Wait            | There exists a set {P0, P1, …, Pn} of waiting processes such that P0 is waiting for a resource that is held by P1, P1 is waiting for a resource that is held by P2, …, Pn–1 is waiting for a resource that is held by Pn, and Pn is waiting for a resource that is held by P0. |

## RAG

Resource Allocation Graph

Directed graph that how processes and resources are related

| Symbol                 | Meaning                                              |               |
| ---------------------- | ---------------------------------------------------- | ------------- |
| Circle                 | Process                                              |               |
| Rectangle with circles | Resource Type with instances                         |               |
| Edges                  | Request Edge (process $P_i$ requests resource $R_j$) | $P_i \to R_j$ |
|                        | Assignment Edge (resource $R_j$ is alloted to $P_i$) | $R_j \to P_i$ |

### Checking deadlock

| Cycle exists? | Instances of each resource type | $\implies$ | Deadlock exists? |
| :-----------: | :-----------------------------: | :--------: | :--------------: |
|       ❌       |               N/A               |            |        ❌         |
|       ✅       |                1                |            |        ✅         |
|       ✅       |            Multiple             |            |   Inconclusive   |

### Disadvantage

Inconclusive for the 3rd case

## Banker’s Algo for Deadlock Detection

It is an algorithm which is implemented as a system process in the OS

In this section, processes are referring to user processes

Let

- $n =$ No of processes
- $m =$ No of resources

|                                     |  Dimension   | Meaning                                                      | Initial value    |
| ----------------------------------- | :----------: | ------------------------------------------------------------ | ---------------- |
| Max<br />Matrix                     | $n \times m$ | Maximum resource **requirement** of each type by every process |                  |
| Allocation<br />Matrix              | $n \times m$ | Total number of resources of each type that is **currently allocated** to each process |                  |
| Need<br />Matrix                    | $n \times m$ | Denotes the number of resources of each resource type that are **yet to be allocated** to a process |                  |
| Available<br />Vector               |     $m$      | **Total** no of instances of each resource type that is **available** |                  |
| Work<br />Vector                    |     $m$      | No of instances of each resource type that is **currently available** | Work = Available |
| Finish<br />Vector                  |     $n$      | Denotes the **completion status** of each process            | False            |
| Request<br />Vector for process $i$ |     $m$      | No of instances of each type of resource that process $i$ requests |                  |

$$
\text{Need}[i][j] = \text{Max}[i][j] - \text{Allocation}[i][j] \\
i \in [1, n],\\
j \in [1, m]
$$

### Vector Comparison

Let $X, Y$ be 2 vectors

$$
X \le Y \iff X[i] \le Y[i], \\
\forall i \in \text{len}(X) = \text{len}(Y) \\
(0, 0, 0) \le (0, 0, 1) \\(0, 1, 0) \not \le (0, 0, 1)
$$

> Every element of $X$ should be smaller than/equal to every corresponding element of $Y$

### Algorithm

1. Initialize all vectors

2. Find process $i$ with

   1. Finish[i] = `False`
   2. Need[i] $\le$ Work

   If we can’t find, go to step 4

3. - Work = Work + Allocation[i]
     - Finish[i] = True
     - Go to step 2

4. If finish[i]==True $\forall i$, the system is in a safe state

### Safe Sequence

You may get ==multiple valid safe sequences== for the same list of process

## Resource-Request Algorithm

Consider a Request[i] vector for process $P_i$

1. Check if Request[i] $\le$ Need[i]

     - if true, got to step 2
     - else, raise error condition that $P_i$ has requested more than it needs

2. Check if Request[i] $\le$ Available[i]

   1. if true, go to step 3
   2. else, $P_i$ must wait, as resources are not available

3. Pretend to allocate requested resources to $P_i$, by modifying the states as follows

   ```
   Available -= Request[i]
   Allocation[i] += Request[i]
   Need[i] -= Request[i]
   ```

4. Run the safety algo to check if the system is in a safe state

     - If safe, resources are allocated to $P_i$
     - else, $P_i$ must wait and the old resource-allocated state is restored

## Deadlock Handling

|                               |      |
| ----------------------------- | ---- |
| Deadlock Avoidance            |      |
| Deadlock Prevention           |      |
| Deadlock Detection & Recovery |      |

## Deadlock Detection

Almost exactly the same as Banker’s; just that this has request instead of need.

1. Initialization

   1. Let `work = available`
   2. Check if $\text{allocation}[i] \ne 0$
      1. true $\implies$ set finish[i] = false
      2. false $\implies$ set finish[i] = true

2. Find $i$ such that

   1. Finish[i] = false
   2. request[i] $\le$ work

   If not found, go to step 4

3. Work = Work + Allocation
   Finish[i] = true
   Go to step 2

6. If finish[i] == false, for some $i \implies$ system is in deadlock state
   or, in other words, there is no deadlock if

     - all the finish[i] = false $\forall i$
     - we are able to derive a safe sequence with all the processes, then there is no deadlock

