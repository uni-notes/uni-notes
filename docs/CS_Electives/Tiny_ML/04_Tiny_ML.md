# TinyML

ML on embedded systems

Often overlooked

## Embedded System

Sensor, Processor, Output all in one small device

### Sensors

IMU: Inertial Measurement Unit

- Accelerometer
- Gyroscope
- Magnetometer

Challenges with IMU sensors

- Interpretability
- Sensor drift: Sensors need to recalibrated regularly

### Processor

MCU: MicroController Unit

Advantages

- Small size
- Low power
- Low cost

## Existing Systems

Nano 33 BLE Sense: AI-enabled ARM-based developmental microcontroller board

![image-20240504184623573](./assets/image-20240504184623573.png)

OV 7675 Camera module

TinyML Shield: Alternative to Breadboard

![image-20240504185015606](./assets/image-20240504185015606.png)

## ARM Cortex Processor Profiles

- ARM designs the processor core & ISA, but they don’t fabricate the chips
- The company (Qualcomm, Apple) bundles it with other design for system-on-chip
- The company (Google, Samsung, etc) places order to fabrication company (TSMC)

![image-20240504185314157](./assets/image-20240504185314157.png)

### Cortex-M ISA

![image-20240504185440541](./assets/image-20240504185440541.png)

## Embedded Systems OS

- RTOS
- Arm MBED OS

## IDK

![image-20240507100908143](./assets/image-20240507100908143.png)

## Memory Usage

- Need to be resource-aware
- Less compute
- Less memory
- Use quantization

![image-20240507100945034](./assets/image-20240507100945034.png)

## TFL Micro

Tensorflow Lite Micro

Built to fit ML on embedded systems

- Very small binary footprint
- No dynamic memory allocation
- No dependencies on complex parts of the standard C/C++ libraries
- No operating system dependencies, can run on bare metal
- Designed to be portable across a wide variety of systems

![image-20240507101050664](./assets/image-20240507101050664.png)

### `g_model`

- Array of bytes
  - Acts as equivalent of a file on disk
- Holds all info about
  - model
  - operators
  - connections
  - trained weights

## Conversion using TFL Micro

- ONIX
- Does not matter if you use PyTorch/Tensorflow

## Hardware & Software Constraints

- OS support
- Compute
- Memory

![image-20240507101607915](./assets/image-20240507101607915.png)

- Long-Running
  - Products are expected to run for months/years which pose challenges for memory allocation
  - Need to guarantee that memory allocation will not end up fragmented
    - Contiguous memory cannot be allocated even if there is enough memory overall

### How TFL micro solves these challenges

- Ask developers to supply a contiguous array of memory ton interpreter
  - The framework avoids any other memory allocations
- Framework guarantees that it won’t allocate from this “arena” after initialization, so long-running applications won’t fail due to to fragmentation
- Ensures
  - clear budget for the memory used by ML
  - framework has no dependency on OS facilities needed by `malloc` or `new`

Size of tensor arena

- Operator variables
- Interpreter state
- Operator I/O

Finding ideal size of arena

- Trial and error
- Create as large an arena as possible
- Use `arena_used_bytes()` to find actual size used
- Resize arena to this length and rebuild
- Best to do this for every deployment platform, since different op implementations may need varying scratch buffer sizes

Ops specification

![image-20240507102746362](./assets/image-20240507102746362.png)

## Workflow

![image-20240507102857295](./assets/image-20240507102857295.png)

## Example

![image-20240507104800241](./assets/image-20240507104800241.png)

