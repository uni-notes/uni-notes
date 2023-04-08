returns

- index, if found
- $-1$, if not found

Space Complexity = $O(1)$

|                      | Linear Search                                | Binary Search                        |
| -------------------- | -------------------------------------------- | ------------------------------------ |
| Working              | Go through each element of array and compare | Divide the array into half each time |
| Worst-Case Time Complexity      | $O(n)$                                       | $O(\log_2 n)$                        |
| Average-Case Time Complexity      | $O(n)$                                       | $O(\log_2 n)$                        |
| Best-Case Time Complexity      | $O(1)$                                       | $O(1)$                        |
| Requires Sorted List | ❌                                            | ✅                                    |

[Search Practicals](Practicals/04_Search.md)
