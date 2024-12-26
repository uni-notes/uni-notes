# Architectures

## Common

```python
class NeuralNet(nn.Module):
	def __init__(self, init_data, hidden_layers):
		# init network architecture
		pass
	def forward(self, x):
		return self.network(self.reshape(x)).squeeze()
	
	# Helper Functions
	def predict_logits(self, X):
		return self.forward(X)
	def predict_proba(self, X):
		return torch.softmax(self.predict_logits(X), dim=0)
	def predict_from_proba(self, proba)
		return proba.argmax(axis=1)
	def predict(self, X):
		return self.predict_from_proba(self.predict_proba(X))
```

## One-vs-Rest Classifier

```python
class NeuralNet(nn.Module):
	def __init__(self, init_data, hidden_layers):
		super().__init__()

		for x, y in DataLoader(init_data):
			self.input_size = x.shape[-1]
			self.output_size = y.shape[-1]
			break
		
		output_layer = nn.LazyLinear(self.output_size) # output layer
		
		layers = (
	  		# [input_layer] +
			hidden_layers +
			[output_layer]
		)

		self.network = nn.Sequential(
			*layers
		)

		# init lazy layers
		self.forward(x)

	def reshape(self, x):
		# batch_size, no_of_channels, width, height
		return x.view(x.shape[0], 1, x.shape[1], x.shape[2])

	def forward(self, x):
		return self.network(self.reshape(x)).squeeze()
```

## One vs Rest with $k-1$ Classifiers

- Advantage: Will save compute if lots of neurons in pre-output layer, which are connected to output layer
- Disadvantage: Looks confusing

```python
class NeuralNet(nn.Module):
	def __init__(self, init_data, hidden_layers):
		super().__init__()

		for x, y in DataLoader(init_data):
			self.input_size = x.shape[-1]
			self.output_size = y.shape[-1]
			break
		
		output_layer = nn.LazyLinear(self.output_size - 1) # output layer
		
		layers = (
	  		# [input_layer] +
			hidden_layers +
			[output_layer]
		)

		self.network = nn.Sequential(
			*layers
		)

		# init lazy layers
		self.forward(x)

	def reshape(self, x):
		# batch_size, no_of_channels, width, height
		return x.view(x.shape[0], 1, x.shape[1], x.shape[2])

	def forward(self, x):
		logits_except_last = self.network(self.reshape(x)).squeeze()
		logit_last = torch.log(1 - torch.exp(logits_except_last).sum())

		logits = (logits_except_last, logit_last.view(-1))
		return logits
```

Testing logic

```python
# Given logits for the first two classes
probs_except_last = torch.tensor([0.1, 0.2])
logits_except_last = probs_except_last.log()

# Compute the logit for the last class
logit_last = torch.log(1 - torch.exp(logits_except_last).sum())

# Combine all logits
logits = torch.cat((logits_except_last, logit_last.view(-1)))

# Compute softmax probabilities
probabilities = torch.softmax(logits, dim=0)

# Verify that probabilities sum to 1
print(f"{probs_except_last = }")
print(f"{logits_except_last = }")
print()
print(f"{probabilities = }")
print(f"{logits = }")
```

```
probs_except_last = tensor([0.1000, 0.2000])
logits_except_last = tensor([-2.3026, -1.6094])

probabilities = tensor([0.1000, 0.2000, 0.7000])
logits = tensor([-2.3026, -1.6094, -0.3567])
```

