# Feature Selection

## MDA

```python
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
 
X = boston["data"]
Y = boston["target"]
 
rf = RandomForestRegressor()
scores = defaultdict(list)
```

```python
from joblib import Parallel, delayed

class FeaturePermutationMetric():
	def __init__(self, estimator, scoring, scoring_lower_is_better, feature_names=None, random_feature_baseline=None, cv=None, random_state=None, n_repeats_fit = 3, n_repeats_feature_permute = 100, n_jobs=None):
		self.estimator = estimator
		self.scoring = scoring
		self.diff_lower_is_better = not scoring_lower_is_better
		self.random_feature_baseline = random_feature_baseline if random_feature_baseline is not None else np.random.normal(loc=0, scale=1, size=X.shape[0])
		self.n_repeats_fit = n_repeats_fit
		self.n_repeats_feature_permute = n_repeats_feature_permute
		self.random_state = random_state
		self.cv = cv if cv is not None else KFold(n_splits=5, random_state=self.random_state)
		self.feature_names = feature_names

		self.n_jobs = n_jobs if n_jobs is not None else 1

		self.comparison_metric = "scoring_diff_after_permuting"
		
	def get_score_after_permute_(self, X, y, train_idx, test_idx):
		score_after_permute_list = []
		
		X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

		for random_state_fit_i in range(0, self.n_repeats_fit, 1):
			random_state_fit = self.random_state + random_state_fit_i
			
			self.estimator_ = clone(self.estimator)
			
			self.estimator_.set_params(random_state=random_state_fit)
			self.estimator_.fit(X_train, y_train)
			
			score_before_permuting = self.scoring(y_test, self.estimator_.predict(X_test))
		
			for random_state_feature_permuted_i in range(0, self.n_repeats_feature_permute, 1):
				random_state_feature_permuted = self.random_state + random_state_feature_permuted_i
				permutation = np.random.default_rng(random_state_feature_permuted).permutation
				
				for feature_permuted in range(X_train.shape[1]):
					X_test_permuted = X_test.copy()
					X_test_permuted[:, feature_permuted] = permutation(X_test_permuted[:, feature_permuted])
					
					score_after_permuting = self.scoring(y_test, self.estimator_.predict(X_test_permuted))
					score_diff_after_permuting = (score_before_permuting - score_after_permuting)
					score_diff_perc_after_permuting = score_diff_after_permuting / score_before_permuting
					
					score_after_permute_list.append(
						[fold, random_state, feature_names[feature_permuted], score_diff_after_permuting, score_diff_perc_after_permuting]
					)
		return score_after_permute_list
		
	def fit(self, X, y):
		if self.feature_names is None:
			try:
				self.feature_names = X.columns
			except:
				raise Exception("No feature names passed")
				pass
		if self.random_feature_baseline:
			self.random_feature_baseline_col_name = "random_feature_baseline"
			feature_names = np.append(feature_names, [random_feature_baseline_col_name])
			X[random_feature_baseline_col_name] = self.random_feature_baseline

		with sklearn.config_context(
			assume_finite = True,
			skip_parameter_validation = True
		):
			cv_results_full = (
				Parallel(n_jobs = self.n_jobs)
				(
					delayed( get_score_after_permute_ )(X, y, train_idx, test_idx)
					for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y))
				)
			)
		self.cv_results_full_ = pd.DataFrame(
			columns = ["fold", "random_state", "feature_name", "score_diff_after_permuting", "score_diff_perc_after_permuting"]
			data = cv_results_full
		)
		
		cv_results_ = (
			self.cv_results_full_
			.groupby("feature_name")
			["score_diff_after_permuting", "score_diff_perc_after_permuting"]
			.agg(["median", "std", "min", "max"])
			.sort(self.comparison_metric, ascending=not self.diff_lower_is_better)
		)
		
		self.important_features_ = self.get_important_features_()
		
		return self
	
	def get_important_features_(self):
		variables_diff = self.cv_results_.query(f"feature_name != {self.random_feature_baseline_col_name}")[self.comparison_metric]["median"]
		random_baseline_diff = self.cv_results_.query(f"feature_name = {self.random_feature_baseline_col_name}")[self.comparison_metric]
		
		random_baseline_diff_best = (
			random_baseline_diff
			[
				"max"
				if not self.diff_lower_is_better
				else "min"
			]
			.iloc[0]
		)
		
		important_features_mask = (
			variables_diff > random_baseline_best_score
			if not self.diff_lower_is_better
			else variables_diff < random_baseline_best_score
		)
		
		return self.cv_results_.eval(important_features_mask)
	
	def transform(self, X, y):
		check_is_fitted(self, "important_features_")
		return X[self.important_features_], y

	def fit_transform(self, X, y):
		return self.fit(X, y).transform(X, y)
```

```python
fpm = FeaturePermutationMetric(
	RandomForest(n_estimators=10, max_depth=5, n_jobs=-1),
	scoring = root_mean_squared_error,
	scoring_lower_is_better = True,
	n_repeats_fit = 3,
	n_repeats_feature_permute = 100,
	n_jobs = -1,
	random_state = 0
)
fpm.fit_transform(X, y)
fpm.cv_results_
```

## Permutation Feature Similarity

```python
def permutation_feature_similarity(X, y, n_estimators=100, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    # Train the random forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    n_features = X.shape[1]
    similarity_matrix = np.zeros((n_features, n_features))
    
    # Permute each feature and measure impact
    for i in range(n_features):
        X_test_permuted = X_test.copy()
        X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
        permuted_accuracy = accuracy_score(y_test, rf.predict(X_test_permuted))
        accuracy_drop = baseline_accuracy - permuted_accuracy
        
        # Calculate impact on other features
        for j in range(n_features):
            if i != j:
                X_test_double_permuted = X_test_permuted.copy()
                X_test_double_permuted[:, j] = np.random.permutation(X_test_double_permuted[:, j])
                double_permuted_accuracy = accuracy_score(y_test, rf.predict(X_test_double_permuted))
                
                # Similarity is inversely proportional to additional accuracy drop
                additional_drop = permuted_accuracy - double_permuted_accuracy
                similarity = 1 - (additional_drop / accuracy_drop) if accuracy_drop > 0 else 0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    return similarity_matrix
```