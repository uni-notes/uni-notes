## Goal of Cache Optimization

Reduce average memory access time by improving the following aspects

| Aspect                                    | Solution                                                     | Advantage               | Disadvantage                                                 |
| ----------------------------------------- | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ |
| Reduce Miss Rate<br />(Increase Hit Rate) | Larger block size (but not too high)                         | Fewer capacity miss     | Longer hit time (due to longer search)<br />Costlier         |
|                                           | Larger cache size                                            |                         |                                                              |
|                                           | Higher Associativity                                         | Fewer conflict miss     | Complicated circuit<br />Longer clock cycle time (increased hit time) |
| Reduce Miss Penalty                       | Multilevel Caches                                            | Reduced turnaround time | Increased overhead of write-back/write-through/write-buffer  |
|                                           | Write-through with buffer to serve reads before writes (to give priority to read misses over writes) |                         |                                                              |
|                                           | Write-Back                                                   |                         |                                                              |
| Reduce Hit Time                           | Use virtually-indexed, physically-tagged, to avoid address translation during indexing of cache |                         |                                                              |
|                                           | Small and Simple caches                                      |                         |                                                              |
|                                           | Pipelined cache access                                       |                         |                                                              |
|                                           | Trace caches                                                 |                         |                                                              |

RAW = Read After Write

## Cache Research Results

### 2:1 cache rule

Miss rate of Direct 

### 8 way Set associate

is as effective as fully associative 

## Cache Segment

Each cache divided into 2 segments

- Instruction Segment
- Data Segment

## Types of Miss Rate

### Local miss rate

$$
\frac{\text{Misses in this cache}}{\text{Number of accesses of this cache}}
$$

### Global miss rate

$$
\frac{\text{Misses in this cache}}{\text{Total number of accesses}}
$$

## Multiple Cache AMAT

$$
\begin{aligned}
\text{AMAT}_\text{Overall} &=
\Big(
\text{Hit Rate}_{L_1} \times \text{Hit Penalty}_{L_1}
\Big) +
\Big(
\text{Miss Rate}_{L_1} \times \textcolor{hotpink}{\text{Miss Penalty}_{L_1}}
\Big) \\
\textcolor{hotpink}{\text{Miss Penalty}_{L_1}} &= 
\Big(
\text{Hit Rate}_{L_2} \times \text{Hit Penalty}_{L_2}
\Big)  +
\Big(
\text{Miss Rate}_{L_2} \times \textcolor{orange}{\text{Miss Penalty}_{L_3}}
\Big) \\
&\dots
\end{aligned}
$$

## Cache Miss Types

| Miss Type                      | When                                                   |
| ------------------------------ | ------------------------------------------------------ |
| Compulsory Miss<br />Cold Miss | Initially caches are empty<br />(Valid bits are all 0) |
| Capacity Miss                  | Not enough space in cache to store                     |
| Conflict Miss                  | Already there is some data in the same cache location  |

