# Persistence

Pickle is not safe

## ONNX

```python
from skl2onnx import to_onnx
onx = to_onnx(clf, X[:1].astype(numpy.float32), target_opset=12)
with open("filename.onnx", "wb") as f:
    f.write(onx.SerializeToString())

from onnxruntime import InferenceSession
with open("filename.onnx", "rb") as f:
    onx = f.read()
sess = InferenceSession(onx, providers=["CPUExecutionProvider"])
pred_ort = sess.run(None, {"X": X_test.astype(numpy.float32)})[0]
```

## Skops

```python
import skops.io as sio
```

```python
# from file
sio.dump(model, "model.skops")

# compression
from zipfile import ZIP_DEFLATED
sio.dump(model, "model.skops", compression=ZIP_DEFLATED, compresslevel=9)

# in-memory
serialized = sio.dumps(model)
```

```python
# from file
unknown_types = get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

# in-memory
unknown_types = get_untrusted_types(serialized)
model = sio.loads(serialized, trusted=unknown_types)
```