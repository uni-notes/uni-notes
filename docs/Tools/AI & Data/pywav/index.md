# PyWavelets

```bash
pip install pywavelets
```

```python
import pywt
import numpy as np
```

```python
# wavelet decomposition
coeffs = pywt.wavedec(y, 'sym5', mode='symmetric')

y_rec = pywt.waverec(coeffs, 'sym5', mode='symmetric')[1:]
```

## Smoothing

```python
def smooth_with_wavelets(y):
    """
    FUNCTION TO SMOOTH SIGNAL VIA WAVELET DECOMPOSITION

    INPUTS:
    - y = array-like signal to smooth

    OUTPUTS:
    - y_rec = smoothed version of input signal

    DEPENDENCIES:
    - PyWavelets 1.3.0
    - numpy 1.21.5
    
    CODE AUTHORED BY: SHAWHIN TALEBI
    """
  

    # wavelet decomposition
    coeffs = pywt.wavedec(y, 'sym5', mode='symmetric')

    # zero out last 5 detail coefficents
    for i in range(5):
        coeffs[i+5] = np.zeros(coeffs[i+5].shape)

    # wavelet recomposition
    y_rec = pywt.waverec(coeffs, 'sym5', mode='symmetric')[1:]

    return y_rec
```

```python
y_smoothed = smooth_with_wavelets(y)
```

```python
# plot result
plt.figure(figsize=(24,8))
plt.rcParams.update({'font.size': 16})
plt.plot(x, y, x, y_smoothed, linewidth=2)
plt.legend(['original', 'smoothed'])
plt.savefig('smoothed_signal_plot.png', facecolor='white')
plt.show()
```

## References

- [ ] https://www.youtube.com/watch?v=gd6oUg608FI
- [ ] https://medium.com/@shouke.wei/how-to-plot-filter-bank-of-a-disctete-wavelet-in-python-42fbb6eb418d
- [ ] https://www.kaggle.com/code/asauve/a-gentle-introduction-to-wavelet-for-data-analysis
