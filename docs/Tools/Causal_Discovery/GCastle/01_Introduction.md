# Introduction

## Custom Test

Mutual information

```python
def mutual_information_fisherz_test(data, x, y, z):
        """Fisher's z-transform for conditional independence test, *using Mutual Information* instead of correlation

        Parameters
        ----------
        data : ndarray
            The dataset on which to test the independence condition.
        x : int
            A variable in data set
        y : int
            A variable in data set
        z : List, default []
            A list of variable names contained in the data set different
            from x and y. This is the separating set that (potentially)
            makes x and y independent.

        Returns
        -------
        _: None
        _: None
        p: float
            the p-value of conditional independence.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(23)
        >>> data = np.random.rand(2500, 4)

        >>> p_value = CITest.fisherz_test(data, 0, 1, [])
        >>> print(p_value)
        0.011609430716781555

        >>> p_value = CITest.fisherz_test(data, 0, 1, [3])
        >>> print(p_value)
        0.01137523908727811

        >>> p_value = CITest.fisherz_test(data, 0, 1, [2, 3])
        >>> print(p_value)
        0.011448214156529746
        """

        n = data.shape[0]
        k = len(z)
        if k == 0:
            # change this
            # r = np.corrcoef(data[:, [x, y]].T)[0][1] 
            pass
        else:
            sub_index = [x, y]
            sub_index.extend(z)
            
            # change this
            # sub_corr = np.corrcoef(data[:, sub_index].T)
            pass
            
            # inverse matrix
            try:
                PM = np.linalg.inv(sub_corr)
            except np.linalg.LinAlgError:
                PM = np.linalg.pinv(sub_corr)
            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
        cut_at = 0.99999
        r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

        # Fisher’s z-transform
        res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
        p_value = 2 * (1 - stats.norm.cdf(abs(res)))

        return None, None, p_value

pc = PC(alpha=0.05, ci_test=test)
```