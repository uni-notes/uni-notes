# ONNX

Open Neural Network eXchange

## PyTorch to TFLite

```python
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from collections import OrderedDict
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare

# Load the trained model from file
trained_dict = torch.load(
	sys.argv[1],
	map_location={'cuda:0': 'cpu'}
)

trained_model = MLP(784, [256, 256], 10)
trained_model.load_state_dict(trained_dict)

if not os.path.exists("%s" % sys.argv[2]):
    os.makedirs("%s" % sys.argv[2])

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "%s/mnist.onnx" % sys.argv[2])

# Load the ONNX file
model = onnx.load("%s/mnist.onnx" % sys.argv[2])

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)
tf_rep.export_graph("%s/mnist.pb" % sys.argv[2])

converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "%s/mnist.pb" % sys.argv[2], tf_rep.inputs, tf_rep.outputs)
tflite_model = converter.convert()

with open("%s/mnist.tflite" % sys.argv[2], "wb") as f:
	f.write(tflite_model)
```