# Basic Model

## Steps

- Define model using `nn.module`
- Cost function
- Optimizer
- Epochs
- Train data
- Predict

## Device

```python
if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

print(f"Using {device}")
```

## Data

### Dataset

```python
class CTDataset(Dataset):
	def __init__(self, filepath, device):
		self.x, self.y = torch.load(filepath, map_location=device)
		self.x = self.x / 255.0
		self.y = nn.functional.one_hot(self.y, num_classes=10).to(float)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, ix):
		return self.x[ix], self.y[ix]
```

```python
# https://www.di.ens.fr/~lelarge/MNIST.tar.gz
train_ds = CTDataset("./MNIST/training.pt", device)
# test_ds = CTDataset('./MNIST/test.pt', device)
```

### Dataloader

```python
train, dev, valid = random_split(train_ds, [0.6, 0.2, 0.2])
```

```python
train_size = min(8, len(train)) # Check if model overfits on small data, to ensure DNN actually is effective
dev_size = min(8, len(dev))

min_training_batches = 4
train_batch_size = min(32, max(1, train_size // min_training_batches))

evaluation_batch_size = min(1_024, dev_size)
```

```python
train_random_sampler = RandomSampler(train, num_samples=train_size)
dev_random_sampler = RandomSampler(dev, num_samples=dev_size)

train_dl = DataLoader(
	train, sampler=train_random_sampler, batch_size=train_batch_size, drop_last=True
)

dev_dl = DataLoader(
	dev, sampler=dev_random_sampler, batch_size=evaluation_batch_size, drop_last=True
)
```

## Model

```python
# architecture
```

### Train

```python
def get_max_len(arrays):
	return max(
		[
			len(array)
			for array
			in arrays
		]
	)

def pad(array, max_len):
	return list(np.pad(
		array,
		pad_width = (0, max_len-len(array)),
		constant_values = np.nan
	))

def get_all_nodes(model):
	network_nodes = []

	layers = model.named_children()
	for i, layer in enumerate(layers):
		layer_nodes_formatted = []
		
		sub_layer = layer[-1]
		for sub_layer_node in sub_layer:
			layer_nodes_formatted.append(sub_layer_node)

		network_nodes.append(layer_nodes_formatted)
	
	return network_nodes

def get_summary_agg(df, agg=["mean"], precision=2):
	df = (
		df
		.groupby(["Epoch", "Train_Time", "Subset"])
		.agg({
			"Loss": agg,
			"Accuracy": ["mean"]
		})
		.round(precision)
	)
	df.columns = list(map('_'.join, df.columns.values))
	df = (
		df
		.reset_index()
		.pivot(
			index=["Epoch", "Train_Time"],
			columns="Subset",
			# values = "Accuracy"
		)
	)
	df.columns = list(map('_'.join, df.columns.values))
	
	# should not be part of data collection
	# df["Generalization_Gap"] = df["Loss_mean_Dev"] - df["Loss_mean_Train"]
	
	df = df.reset_index()
	
	return df
```

```python
# @torch.compile(mode="reduce-overhead")
def train_batch(model, optimizer, loss, x, y, train_dl_len, batch_idx, device, accum_iter=1, k_frac=None):
	x = x.half()
	y = y.half()
	# x = x
	# y = y
	
	model.train()
	# with torch.set_grad_enabled(True): # turn on history tracking
	# forward pass
	proba = model(x)
	loss_array = loss(proba, y)

	loss_scalar = loss_array.mean()
	
	# backward pass
	optimizer.zero_grad(set_to_none=True) # clear accumulated gradients from backpropagation
	loss_scalar.backward()

	# weights update
	# if accum_iter != 1 -> gradient accumulation
	batch_num = batch_idx + 1

	if (
		(batch_num % accum_iter == 0)
		or
		(batch_num == len(train_dl_len))
	):
		optimizer.step()

	

# @torch.compile(mode="reduce-overhead")
def train_epoch(dl, model, optimizer, loss, train_dl_len, device, eval=False, k_frac=None):

	# epoch_accuracies = []
	epoch_losses = []
	epoch_accuracies = []

	for batch_idx, (x, y) in enumerate(dl):
		train_batch(model, optimizer, loss, x, y, train_dl_len, batch_idx, device, accum_iter=1, k_frac=k_frac)
	
		# epoch_accuracies += eval_batch(model, x, y)
		if eval:
			temp = eval_batch(model, x, y, loss, device)
			epoch_losses += temp[0]
			epoch_accuracies += temp[1]

	return epoch_losses, epoch_accuracies

# @torch.compile(mode="reduce-overhead")
def eval_batch(model, x, y, loss, device):
	x = x.half()
	y = y.half()
	
	# x = x
	# y = y

	model.eval()
	with torch.inference_mode(): # turn off history tracking
		# forward pass
		proba = model(x)
		
		loss_value = loss(proba, y)
		epoch_loss_array = loss_value.detach() # loss_value.item() # batch loss

		true = y.argmax(axis=1)
		pred = proba.argmax(axis=1)
		epoch_accuracy_array = (pred == true) # torch.sum()

		return epoch_loss_array, epoch_accuracy_array

# @torch.compile(mode="reduce-overhead")
def eval_epoch(dl, model, loss, device):
	epoch_accuracies = []
	epoch_losses = []
	for batch_idx, (x, y) in enumerate(dl):
		temp = eval_batch(model, x, y, loss, device)
		epoch_losses += temp[0]
		epoch_accuracies += temp[1]

	return epoch_losses, epoch_accuracies


def train_model(train_dl, dev_dl, model, loss, optimizer, n_epochs, device, train_eval_every=10, dev_eval_every=10, agg=None, k_frac=None, log=False):
	print(rf"""
    \n
	Training with {train_dl, dev_dl, model, loss, optimizer, n_epochs, device, train_eval_every, dev_eval_every, agg, k_frac, log}
	""")
	
	model = model.to(device).half()

	model.train()
  
	summary_list = []
  
	train_dl_len = len(train_dl)
 
	print_epoch_every = dev_eval_every

	train_time = 0
	for epoch in range(1, n_epochs + 1):
		print_epoch = False
		eval_train = False
		eval_dev = False
		
		if epoch == 1 or epoch == n_epochs:
			eval_train = True
			eval_dev = True
			if log:
				print_epoch = True
		if epoch % train_eval_every == 0:
			eval_train = True
		if epoch % dev_eval_every == 0:
			eval_dev = True
		if epoch % print_epoch_every == 0:
			print_epoch = True

		if print_epoch:
			print(f"Epoch {epoch}/{n_epochs} started", end="")

		start_time = time.time()
		epoch_train_losses, epoch_train_accuracies = train_epoch(train_dl, model, optimizer, loss, train_dl_len, device, eval=eval_train, k_frac=k_frac)
		end_time = time.time()
		duration = end_time-start_time
		train_time += duration
		
		if eval_dev:
			epoch_dev_losses, epoch_dev_accuracies = eval_epoch(dev_dl, model, loss, device)
		else:
			epoch_dev_losses, epoch_dev_accuracies = [], []
		
		for e, a in zip(epoch_train_losses, epoch_train_accuracies):
			summary_list.append(
				[epoch, train_time,  "Train", float(e), float(a)]
			)
		for e, a in zip(epoch_dev_losses, epoch_dev_accuracies):
			summary_list.append(
				[epoch, train_time, "Dev", float(e), float(a)]
			)

		if print_epoch:
			print(f", completed")

	model.eval()

	summary = (
		 pd.DataFrame(
			columns = ["Epoch", "Train_Time", "Subset", "Loss", "Accuracy"],
			data = summary_list
		)
	)

	if agg is not None:
		summary = summary.pipe(get_summary_agg, agg)

	return summary
```

### Idea

I was watching https://youtu.be/VMj-3S1tku0 and got an idea. I’ve put the same here: https://github.com/karpathy/micrograd/issues/78

#### Context
This is in reference to the step of clearing accumulated gradients at: https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/demo.ipynb#L265

#### Problem

People tend to forget to clear the gradients wrt the loss function backward pass.

#### Idea
Create a way to bind the loss function to the network **_once_**, and then automatically clear accumulated gradients automatically when performing the backward pass.

#### Advantage
We can perform backward pass whenever, wherever, and as many times as we want without worrying about accumulated gradient.

#### Pseudocode

```python
class Loss(Value):
  def __init__(self, bound_network):
    self.bound_network = bound_network

  def __call__(self, batch_size=None):
    # loss function definition
    self.data = data_loss + reg_loss

  def backward():
    # clear gradients of bound network
    bound_network.zero_grad()
    super().backward()    

total_loss = Loss(
  bound_network = model
)

for k in range(100):
  # ...

  # model.zero_grad() # since total_loss is bound to network, it should automatically perform model.zero_grad() before doing the backward
  total_loss.backward()

  # ...
```

#### Questions
1. Is my understanding of the problem correct?
2. Is this change value-adding?
3. Is the above pseudocode logically correct?
4. If the answer to all the above are yes, I could work on a PR with your guidance.

## Loss Curve

```python
def plot_summary(df, x, y):
	df = df.copy()
	c = "Optimizer"
	
	if "Accuracy" in y and "Generalization" not in y:
		sub_title = f"Higher is better"
		percentage = True
	else:
		sub_title = f"Lower is better"
		percentage = False
	
	if percentage:
		df[y] *= 100

	if "Accuracy" in y and "Generalization" not in y:
		range_y = [
			0,
			100
		]
	else:
		range_y = [
			0,
			df[
				df[y] > 0
			][y].quantile(0.90)*1.1
		]

	# if "loss" in y.lower():
	# 	range_y = [0, df[y].quantile(0.90)*1.1]
	# else:
	# 	range_y = None
	# if y == "Generalization_Gap":
	# 	sub_title = f"Lower is better"
	# 	range_y = None
	# else:
	# 	range_y = [0, 100 if percentage else 1]
	# 	sub_title = f"Higher is better"

	title = f'{y.replace("_", " ")}'

	title += f"<br><sup>{sub_title}</sup>"
	
	facet_row = "Train_Batch_Size"

	fig = px.line(
		data_frame=df,
		x=x,
		y=y,
		facet_col="Learning_Rate",
		facet_row="Train_Batch_Size",
		facet_row_spacing = 0.1,
		color = c,
		title = title,
		range_x = [df[x].values.min(), df[x].values.max()],
		range_y = range_y, # df[y].values.min() * 0.95
		markers=True,
	)
	
	n_rows = df[facet_row].unique().shape[0]
	fig.update_layout(height=300*n_rows)
	fig.update_traces(
		patch={
			"marker": {"size": 5},
			"line": {
				"width": 1,
				# "dash": "dot"
			},
		}
	)
	fig.update_traces(connectgaps=True) # required for connecting dev accuracies
	st.plotly_chart(fig, use_container_width=True)
 
	return fig
```

## Multiple Models

```python
import inspect

def train_models(loss, model, n_epochs, optimizer_names, learning_rates, train_batch_sizes, device, agg=["mean"], train_eval_every=10, dev_eval_every=10, log=False, output_path = "summary.csv"):
	# summaries = pd.DataFrame()
	# i=0
	train_size = min(2_048, len(train)) # Check if model overfits on small data, to ensure DNN actually is effective
	dev_size = min(2_048, len(dev))
	train_random_sampler = RandomSampler(train, num_samples=train_size)
	dev_random_sampler = RandomSampler(dev, num_samples=dev_size)
	
	evaluation_batch_size = 2_048
 
	if evaluation_batch_size > dev_size:
		raise Exception("Evaluation batch size > dev size")
	
	for train_batch_size in train_batch_sizes:
		if evaluation_batch_size > train_size:
			raise Exception("Evaluation batch size > dev size")
	
		train_dl = DataLoader(
			train, sampler=train_random_sampler, batch_size=train_batch_size, drop_last=True,
			# num_workers = 1 # 0
		)

		dev_dl = DataLoader(
			dev, sampler=dev_random_sampler, batch_size=evaluation_batch_size, drop_last=True,
			# num_workers = 1 # 0
		)
  
		for learning_rate in learning_rates:
			if learning_rate > 0.0100:
				raise Exception("Very high learning rate")
			for optimizer_name in optimizer_names:
				model_copy = copy.deepcopy(model)
				optimizer = getattr(optim_class, optimizer_name)
				optimizer_kwargs = dict(
					params = model_copy.parameters(),
					lr=learning_rate
				)
				if "eps" in list(inspect.getfullargspec(optimizer.__init__)[0]):
					optimizer_kwargs.update(eps=1e-4)
				optimizer = optimizer(**optimizer_kwargs)
				
				for state in optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = torch.as_tensor(v, device=device).half()
				
				summary = train_model(
					train_dl,
					dev_dl,
					model_copy,
					loss,
					optimizer,
					n_epochs,
					device = device,
					train_eval_every=train_eval_every,
					dev_eval_every=dev_eval_every,
					log=log,
					agg = agg
				)
				summary["Model"] = str(get_all_nodes(model_copy))
				summary["Optimizer"] = optimizer_name
				summary["Learning_Rate"] = learning_rate
				summary["Train_Batch_Size"] = train_batch_size
				
				# disabled due too high space complexity
				# summaries = pd.concat([
				# 	summaries,
				# 	summary
				# ])
				summary.to_csv(
					output_path,
					index = False,
					mode = "a",
					header = not os.path.exists(output_path)
				)
				gc.collect(0)

				# i += 1
				# if i==1:
				# 	break
	
	return None
```

```python
model = NeuralNet(
	init_data = train,
	hidden_layers = [
		nn.Flatten(),
		nn.LazyLinear(10),
		nn.ReLU(),
		# nn.LazyLinear(10),
		# nn.ReLU()
		# nn.Sigmoid() # not required
	]
)
```

```python
def percentile(p):
    def percentile_(x):
        return np.percentile(x, p)
    percentile_.__name__ = f'Percentile_{p}'#.format(n*100)
    return percentile_
```

```python
optimizer_names = [
    # 'ASGD',
    # 'Adadelta',
    # 'Adagrad',
    'Adam',
    # 'AdamW',
    # 'Adamax',
    # # 'LBFGS',
    # 'NAdam',
    # 'RAdam',
    # 'RMSprop',
    # 'Rprop',
    'SGD',
    # 'SparseAdam'
]
```

```python
gc.collect()
gc.set_threshold(0)
summaries = train_models(
  loss = nn.CrossEntropyLoss(reduction="none"),
	model = model,
	n_epochs = 20, # 3
	optimizer_names = optimizer_names,
 	learning_rates = [
      1e-4, 1e-3, 1e-2
    ],
	train_batch_sizes = [
        16, 32, 64
    ],
	device = device,
    agg = [
        "mean",
        # "std",
        # "median",
        percentile(2.5),
        percentile(97.5)
    ],
    train_eval_every=3,
	dev_eval_every=3,
    log = True
)
gc.collect()
gc.set_threshold(g0, g1, g2)
```

