# Custom CV: A Python Library for Specialized Cross-Validation

[![PyPI version](https://badge.fury.io/py/custom-cv.svg)](https://badge.fury.io/py/custom-cv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Custom CV** is a Python library that provides specialized cross-validation strategies for machine learning tasks where standard K-Fold cross-validation is insufficient. It includes robust implementations for stratified, time-series, and spatial data, along with tools for model comparison and evaluation.

This library is designed to be a simple, intuitive, and powerful tool for data scientists and machine learning practitioners who need to perform more rigorous model validation.

## Key Features

- **Stratified K-Fold CV**: Ensures class distribution is preserved in each fold, which is critical for imbalanced datasets.
- **Time-Series CV**: Respects the temporal order of data, ensuring models are always tested on "future" data relative to the training set.
- **Spatial Block CV**: Prevents data leakage from spatial autocorrelation by splitting data into geographic blocks.
- **Statistical Significance Testing**: Includes a paired t-test to determine if the performance difference between two models is statistically significant.
- **Visualizations**: Tools to plot and compare model performance across folds, and to visualize how the data is split by each CV strategy.

## Installation

You can install `custom-cv` directly from PyPI using `pip`:

```bash
pip install custom-cv
```

The package requires Python 3.8 or higher.

## Quickstart Example

Here is a simple example of how to use `TimeSeriesCV` to generate splits and visualize them.

```python
import numpy as np
import matplotlib.pyplot as plt
from custom_cv import TimeSeriesCV, plot_cv_splits

# 1. Create some dummy time-series data
n_samples = 50
X = np.arange(n_samples).reshape(-1, 1)
y = X.flatten() * 2 + np.random.randn(n_samples) * 5

# 2. Initialize our custom time-series splitter
ts_cv = TimeSeriesCV(n_splits=5, test_size=3, gap=2)

# 3. Generate and inspect the splits
print("--- TimeSeriesCV Splits ---")
for fold_num, (train_idx, test_idx) in enumerate(ts_cv.split(X)):
    print(f"Fold {fold_num+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    # ... your model training and evaluation logic would go here ...

# 4. Visualize how the data was split
fig, ax = plot_cv_splits(ts_cv, X, y)
plt.show() # In a script, or it will display automatically in a notebook
```

## Model Comparison Example

You can also use `custom-cv` to compare models and test for statistical significance.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from custom_cv import StratifiedKFoldCV, plot_cv_scores, paired_ttest_cv

# Create data and models
X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.05, random_state=42)
models = {"LR": LogisticRegression(), "RF": RandomForestClassifier(random_state=42)}
cv = StratifiedKFoldCV(n_splits=10)
results = {}

# Evaluate models
for name, model in models.items():
    scores = []
    for train, test in cv.split(X, y):
        model.fit(X[train], y[train])
        scores.append(accuracy_score(y[test], model.predict(X[test])))
    results[name] = scores

# Perform significance test and visualize
t_stat, p_value, is_sig = paired_ttest_cv(results["RF"], results["LR"])
print(f"P-value: {p_value:.4f}, Significant: {is_sig}")

fig, ax = plot_cv_scores(results)
plt.show()
```

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
