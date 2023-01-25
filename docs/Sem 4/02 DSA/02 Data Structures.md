|                          |               Stack               |           Queue           |  Circular Queue   |
| ------------------------ | :-------------------------------: | :-----------------------: | :---------------: |
| Principle                |     LIFO (Last in first out)      | FIFO (First in First out) |       FIFO        |
| Operations               |             Push, Pop             |     Enqueue, Dequeue      | Enqueue, Dequeue  |
| Insertion                |             $t = t+1$             |         $R = R+1$         | $R = (R+1) \% n$  |
| Deletion                 |             $t = t-1$             |         $F = F+1$         | $F = (F+1) \% n$  |
| Size<br />(not capacity) |              $t+ 1$               |         $(R - F)$         | $[n - F+R)] \% n$ |
| Overflow                 |              $t=n-1$              |           $R=n$           |   size $= n-1$    |
| Underflow                |              $t=-1$               |           $F=n$           |    size $= 0$     |
| Time Complexity          |              $O(1)$               |                           |                   |
| Space Complexity         | $O(1 \times \text{element size})$ |                           |                   |

Queue and CQ implementation is different in this course. What we studied in 12th grade is actually better, but we have to follow the textbook.
