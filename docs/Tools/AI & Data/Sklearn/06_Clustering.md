# Clustering

```python
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
import numpy as np

class WarmStartKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters_list, **init_params):
        """
        Custom KMeans estimator with warm start for a list of n_clusters.
        
        Parameters:
        - n_clusters_list: List of integers specifying the number of clusters to try.
        - max_iter: Maximum number of iterations for k-means.
        - tol: Tolerance for convergence.
        - random_state: Random seed for reproducibility.
        """
        self.n_clusters_list = n_clusters_list
        self.init_params = init_params
        self.results_ = {}

    def fit(self, X, y=None, **fit_params):
        """
        Fit the k-means model using warm start for multiple n_clusters values.
        
        Parameters:
        - X: Input data (array-like or sparse matrix).
        - y: Ignored (not used in clustering).
        
        Returns:
        - self: Fitted estimator.
        """
        previous_centroids = None
        
        for i, n_clusters in enumerate(self.n_clusters_list):
            if i == 0:
                # First run: use default 'k-means++' initialization
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    **init_params
                )
            else:
                # Subsequent runs: use centroids from the previous model as initialization
                additional_centroids = np.random.rand(n_clusters - len(previous_centroids), X.shape[1])
                init_centroids = np.vstack([previous_centroids, additional_centroids])
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init=init_centroids,
                    n_init=1,
                    **init_params
                )
            
            # Fit the model and store results
            kmeans.fit(X)
            self.results_[n_clusters] = {
                "model": kmeans,
                "labels": kmeans.labels_,
                "centroids": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_,
            }
            
            # Update previous centroids for warm start
            previous_centroids = kmeans.cluster_centers_
        
        return self

    def predict(self, X):
        """
        Predict cluster labels using the last fitted model.
        
        Parameters:
        - X: Input data (array-like or sparse matrix).
        
        Returns:
        - labels: Cluster labels predicted by the model.
        """
        if not self.results_:
            raise ValueError("The model has not been fitted yet.")
        
        # Use the last fitted model for prediction
        last_model = list(self.results_.values())[-1]["model"]
        return last_model.predict(X)

    def get_results(self):
        """
        Retrieve clustering results for all n_clusters values.
        
        Returns:
        - Dictionary containing models, labels, centroids, and inertia for each n_clusters value.
        """
        return self.results_
```