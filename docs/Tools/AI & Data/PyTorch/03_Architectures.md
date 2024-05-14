# Architectures

## Basic

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

