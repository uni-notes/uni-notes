# Performance Optimization

## Compile

|                    | Type    | Control Flow<br>Supported? |
| ------------------ | ------- | -------------------------- |
| `torch.jit.trace`  | Static  | ❌                          |
| `torch.jit.script` | Static  | ✅                          |
| `torch.compile`    | Dynamic | ✅                          |

### Model
```python
model = NeuralNet()
optimized_model = torch.compile(
	model,
	mode = "reduce-overhead", # "default", "reduce-overhead", "max-autotune"
)
```

### Optimizer

```python
if torch.cuda.get_device_capability() < (7, 0):
    print("torch.compile is not supported on this device.")

@torch.compile(fullgraph=False)
def step(opt):
	opt.step()
```

### Fuse Layers

```python
model = torch.quantization.fuse_modules(
	model,
	[['conv', 'bn', 'relu']],
)
```
### Fuse Operators
```python
@torch.jit.trace # or torch.jit.script or torch.compile
def gelu(x):
	return (
		x * 0.5 *
		(1.0 + torch.erf(x / 1.41421))
	)
```

## Mobile
```python
from torch.utils.mobile_optimizer import optimize_for_mobile

torchscript_model = torch.jit.script(model)

torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torchscript_model_optimized.save("model.pt")
```

## IDK

### GPU

```python
if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

print(f"Using {device}")
```

```python
# tensors
input = input.to(device).half()
output = output.to(device).half()

# model
model = NeuralNet().to(device)
model.half()

# Optimizer
optimizer = SGD(model.parameters(), lr=learning_rate)
for state in optimizer.state.values():
  for k, v in state.items():
    if isinstance(v, torch.Tensor):
      state[k] = v.to(device).half()
```

### IDK

```python
torch.set_num_threads(int) # number of threads used for intraop parallelism
torch.set_num_interop_threads(int) # interop parallelism (e.g. in the JIT interpreter) on the CPU
```

If you have 4 cores and need to do, say, 8 matrix multiplications (with separate data) you could use 4 cores to do each matrix multiplication (intra-op-parallelism). Or you could use a single core for each op and run 4 of them in parallel (inter-op-parallelism).
In training, you also might want to have some cores for the dataloader, for inference, the JIT can parallelize things (I think).
The configuration is documented here, but without much explanation: https://pytorch.org/docs/stable/torch.html#parallelism 1.4k

### IDK

```python
import torch.distributed as dist
dist.init_process_group(backend="gloo")
```

```python
local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
)
```

```python
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(
    ...
    train_data,
    shuffle=False, # train_sampler will shuffle for you.
    sampler=train_sampler,
)
for e in range(1, epochs + 1):
    train_sampler.set_epoch(e) # This makes sure that every epoch the data is distributed to processes differently.
    train(train_loader)
```

```python
# to see which process throws what error
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # do train
    pass

if __name__ == "__main__":
    main()
```

## Quantization

## Precision

```python
torch.set_float32_matmul_precision("high") # "highest", "high", "medium"
```

```python
model.half()
tensor = tensor.half()
```

### AMP
Automatic Mixed Precision

```python
dtype = torch.bfloat16

scaler = (
	torch.GradScaler()
	if (dtype == torch.float16) # Only necessary for FP16
	else None
)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.autocast(device_type = device, dtype=torch.bfloat16):
            scores = model(data)
            loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad(set_to_none=True)
		
		if scaler is None:
	        loss.backward()
	        optimizer.step()
	    else:
		    scaler.scale(loss).backward()
		    scaler.step(optimizer)
		    scaler.update()
```

### IDK

```python
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
# Calibrate your model
def calibrate(model, calibration_data):
    # Your calibration code here
    return model
model = calibrate(model, [])
torch.quantization.convert(model, inplace=True)
```
### IDK

```python
class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x
```

```python
net = VerySimpleNet().to(device)
net.qconfig = torch.ao.quantization.default_qconfig
net.train()

# Insert observers
net_quantized = torch.ao.quantization.prepare_qat(net)
```

```python
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

train_model(model, train_dl, net_quantized, epochs=1)
```

### Quantize the model using the statistics collected

```python
net_quantized.eval()
net_quantized = torch.ao.quantization.convert(net_quantized)
```

## IDK

> Instead of feeding PyTorch sparse tensor directly into the dataloader, I wrote a custom Dataset class which only accept scipy coo_matrix or equivalent. Then, I wrote a custom collate function for the dataloader which to transform scipy coo_matrix to pytorch sparse tensor during data loading.
>
> ~ https://discuss.pytorch.org/t/dataloader-loads-data-very-slow-on-sparse-tensor/117391

```python
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random, 
                          coo_matrix,
                          csr_matrix, 
                          vstack)
from tqdm import tqdm
```

```python
class SparseDataset2():
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 targets:Union[np.ndarray, coo_matrix, csr_matrix], 
                 transform:bool = None):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
        self.transform = transform # Can be removed

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]
      
def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i = torch.LongTensor(indices).to(DEVICE)
    v = torch.FloatTensor(values).to(DEVICE)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).to(DEVICE)
    
def sparse_batch_collate2(batch): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]
    
    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo() # removed vstack
        data_batch = sparse_coo_to_tensor2(data_batch)
    else:
        data_batch = torch.DoubleTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo() # removed vstack
        targets_batch = sparse_coo_to_tensor2(targets_batch)
    else:
        targets_batch = torch.DoubleTensor(targets_batch)
    return data_batch, targets_batch
```

```python
X = random(800000, 300, density=0.25)
y = np.arange(800000)
ds = SparseDataset(X, y)
sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.RandomSampler(ds,
                      generator=torch.Generator(device='cuda')),
    batch_size=1024,
    drop_last=False)
dl = DataLoader(ds, 
                      batch_size = 1, 
                      collate_fn = sparse_batch_collate2,
                      generator=torch.Generator(device='cuda'),
          sampler = sampler)

for x, y in tqdm(iter(dl)):
  pass

# 100%|██████████| 782/782 [00:11<00:00, 71.03it/s]
```

