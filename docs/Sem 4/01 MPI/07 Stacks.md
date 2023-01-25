## Stack

it is a temporary **scratch** memory, for storing variables

memory is segmented into different various segments, and one of them is stack segment

2/4 bytes involved

## Operations

|             | Push                                                 | Pop                                                          |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Direction   | register/memory to stack                             | stack to register/memory                                     |
|             |                                                      | lower register $\leftarrow$ 1st byte<br />higher register $\leftarrow$ 2nd byte |
| Byte        | SP - 1                                               | SP + 1                                                       |
| Word        | SP - 2                                               | SP + 2                                                       |
|             | [SP-1] $\leftarrow$ MSB<br />[SP-2] $\leftarrow$ LSB |                                                              |
| Double Word | SP - 4                                               | SP + 4                                                       |