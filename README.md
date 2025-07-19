# Custom CV: A Custom Cross-Validation Library

**Custom CV** is a Python library that provides specialized cross-validation strategies for machine learning tasks where standard K-Fold cross-validation is insufficient. It includes implementations for stratified, time-series, and spatial data, along with tools for model comparison and evaluation.

This project was built to demonstrate best practices in Python package development, including a clean `src-layout`, automated testing with `pytest`, and distribution via a standard `pyproject.toml` and `setup.py` configuration.

## Key Features

- **Stratified K-Fold CV**: Ensures class distribution is preserved in each fold. Ideal for imbalanced datasets.
- **Time-Series CV**: Respects the temporal order of data, ensuring models are always tested on "future" data relative to the training set.
- **Spatial Block CV**: Prevents data leakage from spatial autocorrelation by splitting data into geographic blocks.
- **Statistical Significance Testing**: Includes a paired t-test to determine if the performance difference between two models is statistically significant.
- **Rich Visualizations**: Tools to plot and compare model performance across folds, and to visualize how the data is split by each CV strategy.

## Installation

First, ensure you have Python 3.8+ installed. It is highly recommended to work within a virtual environment.

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/kiojoel/custom-cv-lib.git
    cd custom-cv-lib
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the Package**
    Install the package in "editable" mode. This allows you to make changes to the source code and have them immediately reflected without reinstalling.
    ```bash
    pip install -e .
    ```

## Quickstart Example

Here's how to use `TimeSeriesCV` to evaluate a model and visualize the splits.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from custom_cv import TimeSeriesCV, plot_cv_splits

# 1. Create dummy time-series data
n_samples = 50
X = np.arange(n_samples).reshape(-1, 1)
y = X.flatten() * 2 + np.random.randn(n_samples) * 5

# 2. Initialize our custom time-series splitter
ts_cv = TimeSeriesCV(n_splits=5, test_size=3, gap=2)

# 3. Use it in a loop (e.g., for model evaluation)
print("--- TimeSeriesCV Splits ---")
for fold_num, (train_idx, test_idx) in enumerate(ts_cv.split(X)):
    print(f"Fold {fold_num+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    # ... your model training and evaluation logic here ...

# 4. Visualize how the data was split
print("\nGenerating split visualization...")
fig, ax = plot_cv_splits(ts_cv, X, y)

# In a script, you must explicitly show or save the plot
# In a Jupyter Notebook, the plot would display automatically
# plt.show()
fig.savefig("timeseries_split_visualization.png")
print("Plot saved to timeseries_split_visualization.png")
```

For more detailed examples covering all features, please see the scripts in the `/examples` directory.

## Running the Test Suite

This project uses `pytest` for unit testing. To run the tests, ensure you have `pytest` installed and run the command from the root directory of the project.

```bash
# Install testing framework
pip install pytest

# Run the tests
pytest
```
