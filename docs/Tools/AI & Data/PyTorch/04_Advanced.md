# Advanced

## Uncertainty

```python
import numpy as np
import torch, torchvision
from torch.autograd import Variable, grad
import torch.distributions as td
import math
from torch.optim import Adam
import scipy.stats


x_data = torch.randn(100)+0.0 ## observed data (here sampled under H0)

N = x_data.shape[0] ## number of observations

mu_null = torch.zeros(1)
sigma_null_hat = Variable(torch.ones(1), requires_grad=True)

def log_lik(mu, sigma):
  return td.Normal(loc=mu, scale=sigma).log_prob(x_data).sum()

## Find theta_null_hat by some gradient descent algorithm (in this case an closed-form expression would be trivial to obtain (see below)):
opt = Adam([sigma_null_hat], lr=0.01)
for epoch in range(2000):
    opt.zero_grad() ## reset gradient accumulator or optimizer
    loss = - log_lik(mu_null, sigma_null_hat) ## compute log likelihood with current value of sigma_null_hat  (= Forward pass)
    loss.backward() ## compute gradients (= Backward pass)
    opt.step()      ## update sigma_null_hat

print(f'parameter fitted under null: sigma: {sigma_null_hat}, expected: {torch.sqrt((x_data**2).mean())}')
#> parameter fitted under null: sigma: tensor([0.9260], requires_grad=True), expected: 0.9259940385818481

theta_null_hat = (mu_null, sigma_null_hat)

U = torch.tensor(torch.autograd.functional.jacobian(log_lik, theta_null_hat)) ## Jacobian (= vector of partial derivatives of log likelihood w.r.t. the parameters (of the full/alternative model)) = score
I = -torch.tensor(torch.autograd.functional.hessian(log_lik, theta_null_hat)) / N ## estimate of the Fisher information matrix
S = torch.t(U) @ torch.inverse(I) @ U / N ## test statistic, often named "LM" (as in Lagrange multiplier), would be zero at the maximum likelihood estimate

pval_score_test = 1 - scipy.stats.chi2(df = 1).cdf(S) ## S asymptocially follows a chi^2 distribution with degrees of freedom equal to the number of parameters fixed under H0
print(f'p-value Chi^2-based score test: {pval_score_test}')
#> p-value Chi^2-based score test: 0.9203232752568568

## comparison with Student's t-test:
pval_t_test = scipy.stats.ttest_1samp(x_data, popmean = 0).pvalue
print(f'p-value Student\'s t-test: {pval_t_test}')
#> p-value Student's t-test: 0.9209265268946605
```

```python
## another example

env_loss = loss_fn(env_outputs, env_targets)
total_loss += env_loss
env_grads = torch.autograd.grad(env_loss, params, retain_graph=True, create_graph=True)

print(env_grads[0])
hess_params = torch.zeros_like(env_grads[0])
for i in range(env_grads[0].size(0)):
    for j in range(env_grads[0].size(1)):
        hess_params[i, j] = torch.autograd.grad(env_grads[0][i][j], params, retain_graph=True)[0][i, j] ##  <--- error here
print(hess_params)
```

## Early-Stopping

Class

```python
  import copy


  class EarlyStopping:
      def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
          self.patience = patience
          self.min_delta = min_delta
          self.restore_best_weights = restore_best_weights
          self.best_model = None
          self.best_loss = None
          self.counter = 0
          self.status = ""

      def __call__(self, model, val_loss):
          if self.best_loss is None:
              self.best_loss = val_loss
              self.best_model = copy.deepcopy(model.state_dict())
          elif self.best_loss - val_loss >= self.min_delta:
              self.best_model = copy.deepcopy(model.state_dict())
              self.best_loss = val_loss
              self.counter = 0
              self.status = f"Improvement found, counter reset to {self.counter}"
          else:
              self.counter += 1
              self.status = f"No improvement in the last {self.counter} epochs"
              if self.counter >= self.patience:
                  self.status = f"Early stopping triggered after {self.counter} epochs."
                  if self.restore_best_weights:
                      model.load_state_dict(self.best_model)
                  return True
          return False
```

Classification

```python
  import time

  import numpy as np
  import pandas as pd
  import torch
  import tqdm
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  from torch import nn
  from torch.autograd import Variable
  from torch.utils.data import DataLoader, TensorDataset

  ## Set random seed for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  def load_data():
      df = pd.read_csv(
          "https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=["NA", "?"]
      )

      le = LabelEncoder()

      x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values
      y = le.fit_transform(df["species"])
      species = le.classes_

      ## Split into validation and training sets
      x_train, x_test, y_train, y_test = train_test_split(
          x, y, test_size=0.25, random_state=42
      )

      scaler = StandardScaler()
      x_train = scaler.fit_transform(x_train)
      x_test = scaler.transform(x_test)

      ## Numpy to Torch Tensor
      x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
      y_train = torch.tensor(y_train, device=device, dtype=torch.long)

      x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
      y_test = torch.tensor(y_test, device=device, dtype=torch.long)

      return x_train, x_test, y_train, y_test, species


  x_train, x_test, y_train, y_test, species = load_data()

  ## Create datasets
  BATCH_SIZE = 16

  dataset_train = TensorDataset(x_train, y_train)
  dataloader_train = DataLoader(
      dataset_train, batch_size=BATCH_SIZE, shuffle=True)

  dataset_test = TensorDataset(x_test, y_test)
  dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

  ## Create model using nn.Sequential
  model = nn.Sequential(
      nn.Linear(x_train.shape[1], 50),
      nn.ReLU(),
      nn.Linear(50, 25),
      nn.ReLU(),
      nn.Linear(25, len(species)),
      nn.LogSoftmax(dim=1),
  )

  model = torch.compile(model,backend="aot_eager").to(device)

  loss_fn = nn.CrossEntropyLoss()  ## cross entropy loss

  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  es = EarlyStopping()

  epoch = 0
  done = False
  while epoch < 1000 and not done:
      epoch += 1
      steps = list(enumerate(dataloader_train))
      pbar = tqdm.tqdm(steps)
      model.train()
      for i, (x_batch, y_batch) in pbar:
          y_batch_pred = model(x_batch.to(device))
          loss = loss_fn(y_batch_pred, y_batch.to(device))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          loss, current = loss.item(), (i + 1) * len(x_batch)
          if i == len(steps) - 1:
              model.eval()
              pred = model(x_test)
              vloss = loss_fn(pred, y_test)
              if es(model, vloss):
                  done = True
              pbar.set_description(
                  f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, {es.status}"
              )
          else:
              pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
```

Regression

```python
  import time

  import numpy as np
  import pandas as pd
  import torch.nn as nn
  import torch.nn.functional as F
  import tqdm
  from sklearn import preprocessing
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  from torch.autograd import Variable
  from torch.utils.data import DataLoader, TensorDataset

  ## Read the MPG dataset.
  df = pd.read_csv(
      "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", na_values=["NA", "?"]
  )

  cars = df["name"]

  ## Handle missing value
  df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

  ## Pandas to Numpy
  x = df[
      [
          "cylinders",
          "displacement",
          "horsepower",
          "weight",
          "acceleration",
          "year",
          "origin",
      ]
  ].values
  y = df["mpg"].values  ## regression

  ## Split into validation and training sets
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=0.25, random_state=42
  )

  ## Numpy to Torch Tensor
  x_train = torch.tensor(x_train, device=device, dtype=torch.float32)
  y_train = torch.tensor(y_train, device=device, dtype=torch.float32)

  x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
  y_test = torch.tensor(y_test, device=device, dtype=torch.float32)


  ## Create datasets
  BATCH_SIZE = 16

  dataset_train = TensorDataset(x_train, y_train)
  dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

  dataset_test = TensorDataset(x_test, y_test)
  dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


  ## Create model

  model = nn.Sequential(
      nn.Linear(x_train.shape[1], 50), 
      nn.ReLU(), 
      nn.Linear(50, 25), 
      nn.ReLU(), 
      nn.Linear(25, 1)
  )

  model = torch.compile(model, backend="aot_eager").to(device)

  ## Define the loss function for regression
  loss_fn = nn.MSELoss()

  ## Define the optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  es = EarlyStopping()

  epoch = 0
  done = False
  while epoch < 1000 and not done:
      epoch += 1
      steps = list(enumerate(dataloader_train))
      pbar = tqdm.tqdm(steps)
      model.train()
      for i, (x_batch, y_batch) in pbar:
          y_batch_pred = model(x_batch).flatten()  #
          loss = loss_fn(y_batch_pred, y_batch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          loss, current = loss.item(), (i + 1) * len(x_batch)
          if i == len(steps) - 1:
              model.eval()
              pred = model(x_test).flatten()
              vloss = loss_fn(pred, y_test)
              if es(model, vloss):
                  done = True
              pbar.set_description(
                  f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]"
              )
          else:
              pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
```

## Dropout

```python
## p = p_drop; NOT p_keep like Tensorflow
model = nn.Sequential(
  nn.Dropout(p=0.2),
  ## ...,
  nn.Dropout(p=0.2),
  ## ...,
)

## make sure to specify model.train() and model.eval(), as dropout processing is only required for "building" the network. no need to processing for evaluation
```

## Quantization

```python
## direct precision reduction
model = model.half() ## changes everything from float32 to float16

## dynamic precision reduction
model = torch.quantization.quantize_dynamic(
  model,
  {torch.nn.Linear},
  dtype=torch.qint8
)

## static precision reduction
## very complicated ngl
```

## Constrained Optimization

Warning: this clamping is not communicated to the optimizer, and in particular destroys the gradients. So the optimizer falsely believes that it has moved the parameter in a certain direction, when in fact it is clamped to the same value as before.

```python
opt = optim.SGD(model.parameters(), lr=0.1)
for i in range(1000):
    out = model(inputs)
    loss = loss_fn(out, labels)
    print(i, loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # enforce the constraint that the weights fall in the range (-1, 1)
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(-1, 1)
```

## Ensembling

### Approach 1

```python
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=10):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(2048+512, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

# Train your separate models
# ...
# We use pretrained torchvision models here
modelA = models.resnet50(pretrained=True)
modelB = models.resnet18(pretrained=True)

# Freeze these models
for param in modelA.parameters():
    param.requires_grad_(False)

for param in modelB.parameters():
    param.requires_grad_(False)

# Create ensemble model
model = MyEnsemble(modelA, modelB)
x = torch.randn(1, 3, 224, 224)
output = model(x)
```

### Approach 2

When you create an ensemble in PyTorch, it's better to use the `nn.ModuleList()` class from PyTorch. The `nn.ModuleList()` has the same functions as a normal Python list like `append()`. When you create an Ensemble Model like this, you can directly call the `backward` operations and the gradient descent will occur through the model.

Below is an ensemble Neural Network (`EnsembleNet`) that uses the `NeuralNet` as individual NN instances for the ensemble.

To use bagging, simply create an X_input_list where the different elements of the list are Tensors that have been sampled with replacement from your training data. (Your X_input_list and the num_ensemble must be of the same size)

You can modify the `EnsembleNet` initialization code to take a list of different neural networks as well.

```python
class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(in_dim, out_dim)
    self.fc2 = nn.Linear(out_dim, 1)


  def forward(self, X):
    """ X must be of shape [-1, in_dim]"""
    X = self.fc1(X)
    return torch.sigmoid(self.fc2(X))


class EnsembleNet(nn.Module):
  def __init__(self, net = NeuralNet, num_ensemble=5, seed_val=SEED):
      super(EnsembleNet, self).__init__()
      self.ensembles = nn.ModuleList()

      for i in range(num_ensemble):
          torch.manual_seed(seed_val*i+1)
          if torch.cuda.is_available(): # To randomize init of NNs for Ensembles
              torch.cuda.manual_seed(seed_val*i+1)
          self.ensembles.append(net)

      self.final = nn.Linear(num_ensemble, 1)

  def forward(self, X_in_list):
      pred = torch.cat([net(X_in_list[i]) for i,net in enumerate(self.ensembles)])
      pred = pred.reshape(-1, len(self.ensembles))
      return torch.sigmoid(self.final(pred))
```

