ISR

Interrupt Service Routine

We need 2 bytes of memory location for pushing the CS contents

Total 6 bytes are required for an interrupt to occur

## Interrupt Vectors

| Interrupt | Physical Address |      |
| --------- | ---------------- | ----- |
| INT 00H       | ${00000}_H$      | ${IP}_0$ |
|           | ${00002}_H$      | ${CS}_0$ |
| INT 01H | ${00004_H}$ | ${IP}_1$ |
|         | ${00006_H}$ | ${CS}_1$ | 
| INT FFH | ${003FC_H}$ | ${IP}_{255}$ | 
|     | ${003FE_H}$ | ${CS}_{255}$ |

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

