# Mapping & Compilation

## Parts of Compiler

- Frontend: Prog lang to IR (Intermediate Representation)
- Middle-end (Optimizer)
- Backend: IR to Assembly/Machine code (Code generator)

## ML Compilation System

![image-20240517125724799](./assets/image-20240517125724799.png)

## Mapping onto Hardware

- Dataflow choice
- Tiling
- Vectorization
- Bind ops to PEs
- On-chip memory management

### Dataflow selection

Compiler can re-order loops without changing functionality

Compiler heuristics can model approximate effect on runtime and memory access

Eg: 1D Convolution![image-20240517130356478](./assets/image-20240517130356478.png)

|                   |                                                              |                                                              |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Weight stationary | ![image-20240517130411411](./assets/image-20240517130411411.png) | ![image-20240517130222461](./assets/image-20240517130222461.png) |
| Output stationary | ![image-20240517130420951](./assets/image-20240517130420951.png) | ![image-20240517181309538](./assets/image-20240517181309538.png) |
| Input stationary  | ![image-20240517130427409](./assets/image-20240517130427409.png) |                                                              |

### Tiling

Choose tile size on which to operate, to fit data in various parts of memory system

Break a loop into nested loops, each of which can be mapped hierarchically onto memory system (DRAM, SRAM, Registers)

![image-20240517181421139](./assets/image-20240517181421139.png)

Other names

- CUDA: Thread Block
- OpenCL: Work Group

![image-20240517181840392](./assets/image-20240517181840392.png)

### Vectorization

Parallelize operations within smallest tile, to leverage hardware parallelism

![image-20240517181854066](./assets/image-20240517181854066.png)

![image-20240517181953782](./assets/image-20240517181953782.png)

### Binding

Specify PE index $i$ that will execute loop iteration $j$

Applicable when no of PEs $\ne$ no of loop iterations

## On-chip memory management

### Graph Compiler

![image-20240517182236188](./assets/image-20240517182236188.png)

### On-Chip buffer

“Spatio-temporal tetris”

Mem management passes:

- Scheduling: order of subgraph execution
- Allocation: where to put data in buffer
- Slicing/fusing: how to break/merge operations

![image-20240517182505772](./assets/image-20240517182505772.png)

![image-20240517182602057](./assets/image-20240517182602057.png)

## Mapping Space

Usually very large

- Many mappings are functionally-identical; eg: binding operations differently
- Many mappings are invalid; eg: tile size doesn’t fit in memory

Navigate space using heuristics

Informed by device performance/energy models

## DLA ISA

- Domain-specific, simple ISAs
- VLIW: Very Long Instruction Word is common

As time progresses, for DLAs, compiler need not worry about loop nests/mapping data flows, as this is all handled in hardware

Operation fusion: Coarse-grained optimizations

![image-20240517183119262](./assets/image-20240517183119262.png)