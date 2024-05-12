# Basic Model

## Steps

- Define model using `nn.module`
- Cost function
- Optimizer
- Epochs
- Train data
- Predict

## Data

### Dataset

```python
class CTDataset(Dataset):
	def __init__(self, filepath):
		self.x, self.y = torch.load(filepath)
		self.x = self.x / 255.0
		self.y = nn.functional.one_hot(self.y, num_classes=10).to(float)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, ix):
		return self.x[ix], self.y[ix]
```

```python
# https://www.di.ens.fr/~lelarge/MNIST.tar.gz
train_ds = CTDataset('./MNIST/training.pt')
# test_ds = CTDataset('./MNIST/test.pt')
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

# @torch.compile(mode="reduce-overhead")
def train_batch(model, optimizer, loss, x, y, train_dl_len, batch_idx, accum_iter=1, k_frac=None):
	# x = x.half()
	# y = y.half()
	
	model.train()
	# with torch.set_grad_enabled(True): # turn on history tracking
	# forward pass
	proba = model(x)
	loss_array = loss(proba, y)

	loss_scalar = loss_array.mean()
	
	# backward pass
	optimizer.zero_grad(set_to_none=True)
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
def train_epoch(dl, model, optimizer, loss, train_dl_len, k_frac=None):

	epoch_accuracies = []
	for batch_idx, (x, y) in enumerate(dl):
		train_batch(model, optimizer, loss, x, y, train_dl_len, batch_idx, accum_iter=1, k_frac=k_frac)
	
		epoch_accuracies += eval_batch(model, x, y)

	return epoch_accuracies

# @torch.compile(mode="reduce-overhead")
def eval_batch(model, x, y):
	# x = x.half()
	# y = y.half()

	model.eval()
	with torch.inference_mode(): # turn off history tracking
		# forward pass
		proba = model(x)
		
		true = y.argmax(axis=1)
		pred = proba.argmax(axis=1)

		epoch_accuracy_array = (pred == true) # torch.sum()

		# epoch_loss_array = loss_value.detach() # loss_value.item() # batch loss

		return epoch_accuracy_array

# @torch.compile(mode="reduce-overhead")
def eval_epoch(dl, model):
	epoch_accuracies = []
	for batch_idx, (x, y) in enumerate(dl):
		epoch_accuracies += eval_batch(model, x, y)

	return epoch_accuracies


def train_model(train_dl, dev_dl, model, loss, optimizer, n_epochs, eval_every=5, k_frac=None, agg=["mean"], log=False):
	model.train()
  
	summary_list = []
  
	train_dl_len = len(train_dl)

	for epoch in range(1, n_epochs + 1):
		epoch_train_accuracies = train_epoch(train_dl, model, optimizer, loss, train_dl_len, k_frac)
		
		if epoch % eval_every == 0 or epoch == 1:
			epoch_dev_accuracies = eval_epoch(dev_dl, model)
		else:
			epoch_dev_accuracies = []
		
		for e in epoch_train_accuracies:
			summary_list.append(
				[epoch, "Train", float(e)]
			)
		for e in epoch_dev_accuracies:
			summary_list.append(
				[epoch, "Dev", float(e)]
			)

		if log:
			print(f"Epoch {epoch}/{n_epochs} Completed")

	model.eval()

	summary = (
	 	pd.DataFrame(
			columns = ["Epoch", "Subset", "Accuracy"],
			data = summary_list
		)
	)
 
	if agg:
		summary = (
			summary
			.groupby(["Epoch", "Subset"])
			.agg(["mean"])
		)
		summary.columns = list(map('_'.join, summary.columns.values))
		summary = (
			summary
			.reset_index()
			.pivot(
				index="Epoch",
				columns="Subset",
				# values = "Accuracy"
			)
		)
		summary.columns = list(map('_'.join, summary.columns.values))
		summary = summary.reset_index()
	return summary
```

```python
model = NeuralNet(
	train_dl,
	hidden_layers = [
		nn.Flatten(),
		nn.LazyLinear(100),
		nn.ReLU(),
		nn.LazyLinear(10),
		nn.ReLU()
		# nn.Sigmoid() not required
	]
)
# model = model.half()
# model = torch.compile(model, mode="reduce-overhead")
```

```python
loss = nn.CrossEntropyLoss(reduction="none")
optimizer = SGD(model.parameters(), lr=0.1)
n_epochs = 20 # 3
```

```python
summary = train_model(
	train_dl,
	dev_dl,
	model,
	loss,
	optimizer,
	n_epochs,
	eval_every=5,
	agg = ["mean"]
)
```

## Loss Curve

```python
def plot_summary(df, percentage=True):
	x = df.columns[0]
	y = df.columns[1:]

	if percentage:
		df[y] *= 100

	fig = px.line(
		data_frame=df,
		x=x,
		y=y,
		title="Loss Curve: Accuracy (Higher is better)",
		range_x=[df[x].values.min(), df[x].values.max()],
		range_y=[0, 100 if percentage else 1], # df[y].values.min() * 0.95
		markers=True,
	)
	fig.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy")
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
 
	fig.show()

plot_summary(summary)
```

## Plot Examples

```python
def plot_examples(data, plot_count=4, fig_size=(10, 5)):
	x, y = data[:plot_count]
	# x = x.half()
	# y = y.half()
	
	pred = model(x).argmax(axis=1)
	cols = 4
	rows = np.ceil(plot_count / cols).astype(int)

	fig, ax = plt.subplots(rows, cols, figsize=fig_size)
	for i in range(plot_count):
		plt.subplot(rows, cols, i + 1)
		plt.imshow(x[i])
		plt.title(f"True: {y[i].argmax()}; Pred: {pred[i]}")
	fig.tight_layout()
	plt.show()

plot_examples(dev, 4)
```

