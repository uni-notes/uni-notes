# Pre/Post Processing

Amdahl’s Law

## Layers of Overhead for DNN

At the application level, “overheads” can take more time than the DNN itself



![image-20240517190014640](./assets/image-20240517190014640.png)

Example: Face Recognition

![image-20240517221006852](./assets/image-20240517221006852.png)

![image-20240517221019695](./assets/image-20240517221019695.png)

![image-20240517221219847](./assets/image-20240517221219847.png)

## Host + Accelerator

![image-20240517221506831](./assets/image-20240517221506831.png)

- Model

  - Sits in CPU main memory

  - Transferred over PCIe to GPU mem

- Input data

  - Arrives over ethernet to CPU
  - Transferred over PCIe to CPU
  - Inference/Training
  - Result send back to CPU over PCIe

Latency impacted by data transfers

### Solution 1: Multiple GPU Cores

![image-20240517222000472](./assets/image-20240517222000472.png)



### Solution 2: Multiple CPU Cores

![image-20240517222202903](./assets/image-20240517222202903.png)

### Solution 3: Dedicated GPU-GPU connections

NVLink

![image-20240517222259936](./assets/image-20240517222259936.png)

### Dedicated FPGA for Packet Processing

![image-20240517222336927](./assets/image-20240517222336927.png)

## Algorithm Codesign Opportunities

![image-20240517230039617](./assets/image-20240517230039617.png)

![image-20240517230056286](./assets/image-20240517230056286.png)

## On-Device Deployment

CPU, GPU, NPU share the same memory on the SOC (System On Chip)

## Mobile-Cloud Inference

![image-20240517230433537](./assets/image-20240517230433537.png)

