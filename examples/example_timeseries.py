import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt


from custom_cv import TimeSeriesCV, plot_cv_splits

def run_timeseries_example():
    """Demonstrates the TimeSeriesCV splitter."""
    print("--- Running TimeSeriesCV Example ---")

    n_samples = 50
    X_time = np.arange(n_samples).reshape(-1, 1)
    y_time = X_time.flatten() * 2 + np.random.randn(n_samples) * 5

    ts_cv = TimeSeriesCV(n_splits=5, test_size=3, gap=2)

    print(f"Splitting {n_samples} samples into {ts_cv.get_n_splits()} folds:")
    model = LinearRegression()
    fold_scores = []

    for fold_num, (train_indices, test_indices) in enumerate(ts_cv.split(X_time)):
        print(f"\nFold {fold_num+1}:")
        print(f"  Train indices: from {train_indices.min()} to {train_indices.max()} ({len(train_indices)} samples)")
        print(f"  Test indices:  from {test_indices.min()} to {test_indices.max()} ({len(test_indices)} samples)")

        X_train, X_test = X_time[train_indices], X_time[test_indices]
        y_train, y_test = y_time[train_indices], y_time[test_indices]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = mean_squared_error(y_test, predictions)
        fold_scores.append(score)

    print("\n--- Final Time-Series Results ---")
    print(f"Average Mean Squared Error: {np.mean(fold_scores):.2f}")

    print("\nGenerating and saving split visualization...")
    fig, ax = plot_cv_splits(ts_cv, X_time)

    if fig:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'timeseries_splits.png')
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_timeseries_example()