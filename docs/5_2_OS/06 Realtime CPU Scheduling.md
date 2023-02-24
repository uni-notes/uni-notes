## Realtime Systems

Real time processes exist along normal processes

Realtime tasks are of higher priority

Latency should be minimized

Due to the above points, we cannot use a general OS system.

In this course, we are assuming that all realtime tasks are [periodic](#periodic task)

## Types

|                              | Soft Realtime System       | Hard Realtime System                                         |
| :--------------------------- | :------------------------- | :----------------------------------------------------------- |
| Strict deadline constraints? | ❌                          | ✅                                                            |
| Deadline miss leads to       | Degradation in performance | failure/destruction                                          |
| Bounded Latency?             | ❌                          | ✅                                                            |
| Example                      | Streaming Video            | Robots in medical treatment<br />Automated chemical plant<br />Auto-missile system |

## Types of Latency

- Interrupt latency
- Dispatch latency
- Event latency

## Periodic Task

Task repeats itself at regular intervals of time

## Execution Time $t$

Time taken for a process to complete execution

## Time Period $p$

(not like time quanta in Round Robin)

The interval at which the process has to repeat itself

## Deadline

Time constraint for execution time

$$
d \le p
$$

In this course, we are **assuming** that

$$
d = p
$$

## CPU/Processor Utilization

$$
\begin{align}
U
&= \frac{t}{\text{p}} \\&= \frac{\text{Execution Time}}{\text{Period}}
\end{align}
$$

If you have multiple tasks $T_1, T_2, T_3, \dots, T_n$

$$
\begin{align}
U &= U_1 + U_2 + \dots + U_n \\
&= \frac{t_1}{p_1} + \frac{t_2}{p_2} + \dots + \frac{t_n}{p_n}
\end{align}
$$

## RMS/RMA

Rate-Monotic Scheduling Algorithm

Schedule periodic tasks

**Pre-emptive scheduling**

Priority-Based Algo

- Assigns **static priorities** to tasks
- ==**shorter periods means higher priority**==

CPU utilization is not always 100%. It is bounded to a limit, based on number of tasks in system

### Assumtions

- $d=p$
- $t = \text{constant}$

### Given

- Task set with $n$ tasks
- uni-processor

### Todo

Find if task set is schedulable using

- [Test of Schedulability](#Test of Schedulability)
- [Test of Maximum CPU Utilization Bound](#Test of Maximum CPU Utilization Bound)

### Maximum CPU Utilization Bound

For $n$ tasks

$$
U_\text{max} = n(2^\frac{1}{n} - 1)
$$

### 3 Possible Outcomes

|       $U_\text{tot}$        |                      Schedulable?                      |
| :-------------------------: | :----------------------------------------------------: |
|            $> 1$            |                           ❌                            |
|     $\le U_\text{max}$      |                           ✅                            |
| $U_\max < U_\text{tot} < 1$ | Inconclusive<br />(Draw graph to check schedulability) |

## Earliest Deadline First

Priorities of tasks changes dynamically

- Earlier Deadline $\to$ High Priority
- Later Deadline $\to$ Low Priority

### Requirement

Tasks must announce their deadlines to scheduler, when it becomes runnable

[Test of Schedulability](#Test of Schedulability) is the only Necessary & Sufficient Condition

## Test of Schedulability

Check if $U_\text{tot} \le 1$

## Test of Maximum CPU Utilization Bound

Also called as **upper bound of schedulability test**

Check if $U_\text{tot} \le U_\text{max}$

- This is a **sufficient, but not necessary** condition
- $U_\text{max}$ is called as
    - Upper bound of schedulability
    - Maximum CPU utilization bound

## Realtime Scheduling Gantt Chart

If the total time to plot is not given, plot till the LCM of the periods of the processes.

## Priority-Based Pre-emptive Scheduler

Foreground-Background Scheduler

Realtime task are given higher priority than low priority

$$
T_B = \frac{t_B}{
1 - \sum_{i=1}^n somethign
}
$$

