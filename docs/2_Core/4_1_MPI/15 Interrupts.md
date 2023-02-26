ISR

Interrupt Service Routine

We need 2 bytes of memory location for pushing the CS contents

Total 6 bytes are required for an interrupt to occur

## Interrupt Vectors

| Interrupt | Physical Address |       |
| --------- | ---------------- | ----- |
| 00H       | ${00000}_H$      | IP~0~ |
|           | ${00002}_H$      | CS~0~ |
| FFH       |                  |       |

## Interrupts

| `INT` | Interrupt                  | When                                | Explanation                                                  |
| ----- | -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| 0     | Divide by Zero             |                                     |                                                              |
| 1     | Single Step                |                                     |                                                              |
| 2     | NMI                        | low-to-high transition on NMI input | Type 2 interrupts cannot be disabled(masked) by any instruction |
| 3     | **BreakPoint**             |                                     |                                                              |
| 4     | **into**                   |                                     |                                                              |
| 5     | `bound`                    |                                     |                                                              |
| 6     | Invalid opcode             |                                     |                                                              |
| 7     | Co-Processor not available |                                     |                                                              |
| 8     | Double Fault               |                                     |                                                              |
| 9     |                            |                                     |                                                              |
| A     |                            |                                     |                                                              |
| B     |                            |                                     |                                                              |
| C     |                            |                                     |                                                              |
| D     |                            |                                     |                                                              |
| E     |                            |                                     |                                                              |
| F     |                            |                                     |                                                              |

