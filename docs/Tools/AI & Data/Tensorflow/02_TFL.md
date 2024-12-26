# TensorFlow Lite

## Quantization-Aware Training

```python
from tensorflow_model_optimization.quantization.keras import quantize_model

q_aware_model = quantize_model(model) # untrained model

q_aware_model.compile(
	# ...
)

q_aware_model.fit()
q_aware_model.evaluate()
q_aware_model.predict()

# perform post-training optimization
```

## Post-Training Optimization

```python
tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ... optimization

tflite_model = tf_lite_converter.convert()

with open("model.tflite", "wb") as f:
	f.write(tflite_model)
```

### Quantization

```python
tf_lite_converter.target_spec.supported_types = [
	tf.int8
]
tf_lite_converter.target_spec.supported_ops = [
	tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
tf_lite_converter.optimizations = [
	# tf.lite.Optimize.DEFAULT,
	tf.lite.Optimize.OPTIMIZE_FOR_SIZE,
	# tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
]

def representative_data_gen():
	for input_value, _ in test_batches.take(100):
		yield [input_value]

tf_lite_converter.representative_dataset = representative_data_gen
```

## Evaluating model

Testing the model without edge device

```python
interpreter = tf.lite.Interpreter(model_content = tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)
```

```python
x_new = np.array(
	[
		[10.0, 5.0],
		[5.0, 5.0],
	],
	dtype=np.float32
)

interpreter.set_tensor(
	input_details[0]["index"],
	x_new
)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[0]["index"])
print(tflite_results)
```

