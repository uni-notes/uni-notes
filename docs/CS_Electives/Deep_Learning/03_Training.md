# IDK

## Steps

- Forward pass
- Backward pass
- Weights update

## Batching

When training a neural network, we usually divide our data in mini-batches and go through them one by one. The network predicts batch labels, which are used to compute the loss with respect to the actual targets. Next, we perform backward pass to compute gradients and update model weights in the direction of those gradients.

- Full dataset does not fit in memory
- Faster convergence due to stochasticity

## Worst-First Backpropagation

Backpropagation is expensive, so only focus on Top k

## Gradient Accumulation

- Use a small batch size
- Save the gradients at each batch
- Update network weights once every couple of batches

Purpose

- Helps to imitate a larger batch size
- For large GPU memory intensive architectures

Notes

- Some network architectures have batch-specific operations. For instance, batch normalization is performed on a batch level and therefore may yield slightly different results when using the same effective batch size with and without gradient accumulation
- It is important to also update weights on the last batch, to ensure that the last batches are not discarded and used for optimizing the network

## Performance Improvement

### 1. Consider using another learning rate schedule

The learning rate (schedule) you choose has a large impact on the speed of convergence as well as the generalization performance of your model.

Cyclical Learning Rates and the 1Cycle learning rate schedule are both methods introduced by Leslie N. Smith ([here](https://arxiv.org/pdf/1506.01186.pdf) and [here](https://arxiv.org/abs/1708.07120)), and then popularised by fast.ai's Jeremy Howard and Sylvain Gugger ([here](https://www.fast.ai/2018/07/02/adam-weight-decay/) and [here](https://github.com/sgugger/Deep-Learning/blob/master/Cyclical LR and momentums.ipynb)). Essentially, the 1Cycle learning rate schedule looks something like this:



![r/MachineLearning - [D] Here are 17 ways of making PyTorch training faster – what did I miss?](./assets/sc37u5knmxa61.png)

Sylvain writes:

> [1cycle consists of]  two steps of equal lengths, one going from a lower learning rate to a higher one than go back to the minimum. The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower. Then, the length of this cycle should be slightly less than the total number of epochs, and, in the last part of training, we should allow the learning rate to decrease more than the minimum, by several orders of magnitude.

In the best case this schedule achieves a massive speed-up – what Smith calls *Superconvergence* – as compared to conventional learning rate schedules. Using the 1Cycle policy he needs ~10x fewer training iterations of a ResNet-56 on ImageNet to match the performance of the original paper, for instance). The schedule seems to perform robustly well across common architectures and optimizers.

PyTorch implements both of these methods `torch.optim.lr_scheduler.CyclicLR` and `torch.optim.lr_scheduler.OneCycleLR,` see [the documentation](https://pytorch.org/docs/stable/optim.html).

One drawback of these schedulers is that they introduce a number of additional hyperparameters. [This post](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8) and [this repo](https://github.com/davidtvs/pytorch-lr-finder), offer a nice overview and implementation of how good hyper-parameters can be found including the Learning Rate Finder mentioned above.

Why does this work? It doesn't seem entirely clear but one[ possible explanation](https://arxiv.org/pdf/1506.01186.pdf) might be that regularly increasing the learning rate helps to traverse [saddle points in the loss landscape ](https://papers.nips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf)more quickly.

### 2. Use multiple workers and pinned memory in DataLoader

When using [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), set `num_workers > 0`, rather than the default value of 0, and `pin_memory=True`, rather than the default value of False. Details of this are [explained here](https://pytorch.org/docs/stable/data.html).

[Szymon Micacz](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf) achieves a 2x speed-up for a single training epoch by using four workers and pinned memory.

A rule of thumb that [people are using ](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)to choose the number of workers is to set it to four times the number of available GPUs with both a larger and smaller number of workers leading to a slow down.

Note that increasing num_workerswill increase your CPU memory consumption.

### 3. Max out the batch size

This is a somewhat contentious point. Generally, however, it seems like using the largest batch size your GPU memory permits will accelerate your training (see [NVIDIA's Szymon Migacz](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf), for instance). Note that you will also have to adjust other hyperparameters, such as the learning rate, if you modify the batch size. A rule of thumb here is to double the learning rate as you double the batch size.

[OpenAI has a nice empirical paper](https://arxiv.org/pdf/1812.06162.pdf) on the number of convergence steps needed for different batch sizes. [Daniel Huynh](https://towardsdatascience.com/implementing-a-batch-size-finder-in-fastai-how-to-get-a-4x-speedup-with-better-generalization-813d686f6bdf) runs some experiments with different batch sizes (also using the 1Cycle policy discussed above) where he achieves a 4x speed-up by going from batch size 64 to 512.

[One of the downsides](https://arxiv.org/pdf/1609.04836.pdf) of using large batch sizes, however, is that they might lead to solutions that generalize worse than those trained with smaller batches.

### 4. Use Automatic Mixed Precision (AMP)

The release of PyTorch 1.6 included a native implementation of Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.

In the best case, the usage of AMP would look something like this:

```
import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

for data, label in data_iter:
   optimizer.zero_grad()
   # Casts operations to mixed precision
   with torch.cuda.amp.autocast():
      loss = model(data)

   # Scales the loss, and calls backward()
   # to create scaled gradients
   scaler.scale(loss).backward()

   # Unscales gradients and calls
   # or skips optimizer.step()
   scaler.step(optimizer)

   # Updates the scale for next iteration
   scaler.update()
```

Benchmarking a number of common language and vision models on NVIDIA V100 GPUs, [Huang and colleagues find](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) that using AMP over regular FP32 training yields roughly 2x – but upto 5.5x – training speed-ups.

Currently, only CUDA ops can be autocast in this way. See the [documentation](https://pytorch.org/docs/stable/amp.html#op-eligibility) here for more details on this and other limitations.

[u/SVPERBlA](https://www.reddit.com/user/SVPERBlA/) points out that you can squeeze out some additional performance (~ 20%) from AMP on NVIDIA Tensor Core GPUs if you convert your tensors to the [Channels Last memory format](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html). Refer to [this section](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout) in the NVIDIA docs for an explanation of the speedup and more about NCHW versus NHWC tensor formats.

### 5. Consider using another optimizer

AdamW is Adam with weight decay (rather than L2-regularization) which was popularized by fast.ai and is now available natively in PyTorch as `torch.optim.AdamW`. AdamW seems to consistently outperform Adam in terms of both the error achieved and the training time. See [this excellent blog](https://www.fast.ai/2018/07/02/adam-weight-decay/) post on why using weight decay instead of L2-regularization makes a difference for Adam.

Both Adam and AdamW work well with the 1Cycle policy described above.

There are also a few not-yet-native optimizers that have received a lot of attention recently, most notably LARS ([pip installable implementation](https://github.com/kakaobrain/torchlars)) and [LAMB](https://github.com/cybertronai/pytorch-lamb).

NVIDA's APEX implements fused versions of a number of common optimizers such as [Adam](https://nvidia.github.io/apex/optimizers.html). This implementation avoid a number of passes to and from GPU memory as compared to the PyTorch implementation of Adam, yielding speed-ups in the range of 5%.

### 6. Turn on cudNN benchmarking

If your model architecture remains fixed and your input size stays constant, setting `torch.backends.cudnn.benchmark = True` might be beneficial ([docs](https://pytorch.org/docs/stable/backends.html#torch-backends-cudnn)). This enables the cudNN autotuner which will benchmark a number of different ways of computing convolutions in cudNN and then use the fastest method from then on.

For a rough reference on the type of speed-up you can expect from this, [Szymon Migacz](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf) achieves a speed-up of 70% on a forward pass for a convolution and a 27% speed-up for a forward + backward pass of the same convolution.

One caveat here is that this autotuning might become very slow if you max out the batch size as mentioned above.

### 7. Beware of frequently transferring data between CPUs and GPUs

Beware of frequently transferring tensors from a GPU to a CPU using `tensor.cpu()` and vice versa using `tensor.cuda()` as these are relatively expensive. The same applies for `.item()` and `.numpy()` – use `.detach()` instead.

If you are creating a new tensor, you can also directly assign it to your GPU using the keyword argument `device=torch.device('cuda:0')`.

If you do need to transfer data, using `.to(non_blocking=True)`, might be useful [as long as you don't have any synchronization points](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4) after the transfer.

If you really have to, you might want to give Santosh Gupta's [SpeedTorch](https://github.com/Santosh-Gupta/SpeedTorch) a try, although it doesn't seem entirely clear when this actually does/doesn't provide speed-ups.

### 8. Use gradient/activation checkpointing

Quoting directly from the [documentation](https://pytorch.org/docs/stable/checkpoint.html):

> Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does **not** save intermediate activations, and instead recomputes them in backward pass. It can be applied on any part of a model.
>
> Specifically, in the forward pass, function will run in [torch.no_grad()](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad) manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the functionparameter. In the backwards pass, the saved inputs and function is retrieved, and the forward pass is computed on function again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

So while this will might slightly increase your run time for a given batch size, you'll significantly reduce your memory footprint. This in turn will allow you to further increase the batch size you're using allowing for better GPU utilization.

While checkpointing is implemented natively as `torch.utils.checkpoint`([docs](https://pytorch.org/docs/stable/checkpoint.html)), it does seem to take some thought and effort to implement properly. Priya Goyal [has a good tutorial ](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)demonstrating some of the key aspects of checkpointing.

### 9. Use gradient accumulation

Another approach to increasing the batch size is to accumulate gradients across multiple `.backward()` passes before calling optimizer.step().

Following [a post](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) by Hugging Face's Thomas Wolf, gradient accumulation can be implemented as follows:

```
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulate
```

This method was developed mainly to circumvent GPU memory limitations and I'm not entirely clear on the trade-off between having additional `.backward()` loops. [This discussion](https://forums.fast.ai/t/accumulating-gradients/33219/28) on the fastai forum seems to suggest that it can in fact accelerate training, so it's probably worth a try.

### 10. Use Distributed Data Parallel for multi-GPU training

Methods to accelerate distributed training probably warrant their own post but one simple one is to use `torch.nn.DistributedDataParallel` rather than `torch.nn.DataParallel`. By doing so, each GPU will be driven by a dedicated CPU core avoiding the GIL issues of DataParallel.

In general, I can strongly recommend reading the [documentation on distributed training.](https://pytorch.org/tutorials/beginner/dist_overview.html)

### 11. Set gradients to None rather than 0

Use `.zero_grad(set_to_none=True)` rather than `.zero_grad()`.

Doing so will let the memory allocator handle the gradients rather than actively setting them to 0. This will lead to yield a *modest* speed-up as they say in the [documentation](https://pytorch.org/docs/stable/optim.html), so don't expect any miracles.

Watch out, doing this is not side-effect free! Check the docs for the details on this.

### 12. Use .as_tensor() rather than .tensor()

`torch.tensor()` always copies data. If you have a numpy array that you want to convert, use `torch.as_tensor()` or `torch.from_numpy()` to avoid copying the data.

### 13. Turn on debugging tools only when actually needed

PyTorch offers a number of useful debugging tools like the [autograd.profiler](https://pytorch.org/docs/stable/autograd.html#profiler), [autograd.grad_check](https://pytorch.org/docs/stable/autograd.html#numerical-gradient-checking), and [autograd.anomaly_detection](https://pytorch.org/docs/stable/autograd.html#anomaly-detection). Make sure to use them to better understand when needed but to also turn them off when you don't need them as they will slow down your training.

### 14. Use gradient clipping

Originally used to avoid exploding gradients in RNNs, there is both some [empirical evidence as well as some theoretical support](https://openreview.net/forum?id=BJgnXpVYwS) that clipping gradients (roughly speaking: `gradient = min(gradient, threshold)`) accelerates convergence.

Hugging Face's [Transformer implementation](https://github.com/huggingface/transformers/blob/7729ef738161a0a182b172fcb7c351f6d2b9c50d/examples/run_squad.py#L156) is a really clean example of how to use gradient clipping as well as some of the other methods such as AMP mentioned in this post.

In PyTorch this can be done using `torch.nn.utils.clip_grad_norm_`([documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_)).

It's not entirely clear to me which models benefit how much from gradient clipping but it seems to be robustly useful for RNNs, Transformer-based and ResNets architectures and a range of different optimizers.

### 15. Turn off bias before BatchNorm

This is a very simple one: turn off the bias of layers before BatchNormalization layers. For a 2-D convolutional layer, this can be done by setting the bias keyword to False: `torch.nn.Conv2d(..., bias=False, ...)`.  (Here's a r[eminder why this makes sense](https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers).)

You will save some parameters, I would however expect the speed-up of this to be relatively small as compared to some of the other methods mentioned here.

### 17. Use input and batch normalization

You're probably already doing this but you might want to double-check:

- Are you [normalizing](https://pytorch.org/docs/stable/torchvision/transforms.html) your input?
- Are you using [batch-normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)?

And [here's](https://stats.stackexchange.com/questions/437840/in-machine-learning-how-does-normalization-help-in-convergence-of-gradient-desc) a reminder of why you probably should.

#### Bonus tip from the comments: Use JIT to fuse point-wise operations.

If you have adjacent point-wise operations you can use [PyTorch JIT](https://pytorch.org/docs/stable/jit.html#creating-torchscript-code) to combine them into one FusionGroup which can then be launched on a single kernel rather than multiple kernels as would have been done per default. You'll also save some memory reads and writes.

[Szymon Migacz shows](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf) how you can use the `@torch.jit.script` decorator to fuse the operations in a GELU, for instance:

```
@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
```

In this case, fusing the operations leads to a 5x speed-up for the execution of `fused_gelu`
as compared to the unfused version.

See also [this post](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/) for an example of how Torchscript can be used to accelerate an RNN.

Hat tip to [u/Patient_Atmosphere45](https://www.reddit.com/user/Patient_Atmosphere45/) for the suggestion.

## Regularization

### Dropout

#### Regular Dropout

May lead to missing relevant information, since sequential part may involve variable-length inputs

#### Variational Dropout

IDK

### Zoneout

Skip hidden state update and keep the same as previously during training

$$
h_t = h_{t−1}
$$

![image-20230527201100703](./assets/image-20230527201100703.png)

- Robustness against skipping observations in sequence
- Robustness of state representation relative to hidden state updates

### Parameter Averaging

Train RNN and average weights over run

### Stochastic Weight Averaging

Parameter averaging + Continuously varying learning rate

### Fraternal Dropout

Dropout while minimizing variation between outputs to increase robustness to parameterization

## Gradient Problems

FFNN can cope with these problems because they only have a few hidden layers, but RNN struggles.

|                                                              | Vanishing (Converging) | Exploding (Diverging) |
| ------------------------------------------------------------ | ---------------------- | --------------------- |
| Cause<br />Weights multiplied during BPTT are                | Too small              | Too large             |
| Gradients __ exponentially during back-propagation           | shrink                 | grow                  |
| Resultant problem<br />Effect on current output due to past input | Too little             | Too high              |
| Solutions                                                    | Scaling                | Clipping              |

### Initial Weights

We can avoid this by initializing the weights very carefully

### Clipping

rescales gradient to size at most $\theta$.

$$
g \leftarrow \min \left( 1, \frac{\theta}{\vert g \vert}  \right) g
$$

If the weights are large, the gradients grow exponentially during back-propagation

