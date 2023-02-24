## Realtime Systems

Realtime processes exist along normal processes, and their tasks are of higher priority $\implies$ Latency should be minimized

In this course, we are assuming that all realtime tasks are periodic, ie task repeats itself at regular intervals of time

## Types of Realtime Systems

|                              |    Soft Realtime System    |                     Hard Realtime System                     |
| :--------------------------- | :------------------------: | :----------------------------------------------------------: |
| Strict deadline constraints? |             ❌              |                              ✅                               |
| Deadline miss leads to       | Degradation in performance |                     failure/destruction                      |
| Bounded Latency?             |             ❌              |                              ✅                               |
| Example                      |      Streaming Video       | Robots in medical treatment<br />Automated chemical plant<br />Auto-missile system |

## Types of Latency

- Interrupt latency
- Dispatch latency
- Event latency

## Terms

| Term                  | Symbol | Meaning                                                      |
| --------------------- | :----: | ------------------------------------------------------------ |
| Execution Time        |  $t$   | Time taken for a process to complete execution               |
| Time Period           |  $p$   | The interval at which the process has to repeat itself<br/>(not like time quanta in Round Robin) |
| Deadline              |  $d$   | Time constraint for execution time $:d \le p$<br />In this course, we are assuming that $d = p$ |
| Processor Utilization |  $U$   | Fraction of utilization of available processor resources<br />$U = \sum\limits_{i=1}^n \frac{t}{p}$, where $n=$ number of tasks |

## Scheduling Algorithms

Algorithms to complete a set of $n$ tasks, using a single processor, such that

- $d=p$
- $t =$ constant

CPU utilization is not always 100%. It is bounded to a limit, based on number of tasks in system

|                                |                           RMS/RMA                            | EDF                                                          |
| ------------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------- |
| Full Form                      |              Rate-Monotic Scheduling Algorithm               | Earliest Deadline First                                      |
| Priority-Based                 |                              ✅                               | ✅                                                            |
| Priority Type                  |                            Static                            | Dynamic                                                      |
| High Priority for task with __ |                       Shortest Period                        | Earliest Deadline                                            |
| Schedulability Condition(s)    | - [Test of Schedulability](#Test of Schedulability)<br />- [Test of Maximum CPU Utilization Bound](#Test of Maximum CPU Utilization Bound) | - [Test of Schedulability](#Test of Schedulability) (Necessary & Sufficient Condition) |
| Requirement                    |                                                              | Tasks must announce their deadlines to scheduler, when it becomes runnable |

### 3 Cases

|       $U_\text{tot}$        |                       RMS Schedulable?                       | EDF Schedulable? |
| :-------------------------: | :----------------------------------------------------------: | :--------------: |
|            $> 1$            |                              ❌                               |        ❌         |
|     $\le U_\text{max}$      |                              ✅                               |        ✅         |
| $U_\max < U_\text{tot} < 1$ | Inconclusive<br />(Draw [Realtime Scheduling Gantt Chart](#Realtime Scheduling Gantt Chart)) |        ✅         |

## Testing Methods

| Test                                                         | Check                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Test of Schedulability                                       | $U_\text{tot} \le 1$                                         |
| Test of Maximum CPU Utilization Bound<br />(aka upper bound of schedulability test) | $U_\text{tot} \le n(2^\frac{1}{n} - 1)$                      |
| Realtime Scheduling Gantt Chart                              | Scheduling the task set using a Gantt chart<br />(If the total time to plot is not given, plot till the LCM of the periods of the processes.) |
