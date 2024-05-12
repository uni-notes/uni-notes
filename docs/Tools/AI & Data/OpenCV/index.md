# OpenCV

```python
import cv2 as cv
improt numpy as np
import matplotlib.pyplot as plt
```

## Read Image

```python
img = cv.imread(
  "/fruits.jpg",
  cv.IMREAD_GRAYSCALE # COLOR, GRAYSCALE, UNCHANGED
) # numpy array
```

```python
# x, y, channel
img[:, :, 0] # particular channel only
```

## Show

```python
plt.axis("off")
plt.imshow(img)
plt.show()
```

## Filters

```python
img = cv.cvtColor(
  img,
  cv.COLOR_BGR2RGB
)
```

## Export

```python
cv.imwrite(
	"output.png",
  img
)
```

