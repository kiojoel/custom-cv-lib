import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_cv import SpatialBlockCV, plot_cv_splits

def run_spatial_example():
    """Demonstrates the SpatialBlockCV splitter and its visualization."""
    print("--- Running SpatialBlockCV Example ---")

    np.random.seed(42)
    n_samples = 200
    X_spatial = np.random.rand(n_samples, 5)
    coords = np.random.rand(n_samples, 2) * 100
    y_spatial = coords[:, 0] + coords[:, 1] + np.random.randn(n_samples) * 5

    spatial_cv = SpatialBlockCV(n_blocks_per_dim=4)

    print(f"Splitting {n_samples} spatial samples using a {spatial_cv.n_blocks_per_dim}x{spatial_cv.n_blocks_per_dim} grid.")

    fold_num = 0
    for train_indices, test_indices in spatial_cv.split(X=X_spatial, y=y_spatial, coords=coords):
        fold_num += 1
        test_coords = coords[test_indices]
        min_x, min_y = test_coords.min(axis=0)
        max_x, max_y = test_coords.max(axis=0)

        print(f"\nFold {fold_num}: Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        print(f"  Test points are in x range: [{min_x:.1f}, {max_x:.1f}], y range: [{min_y:.1f}, {max_y:.1f}]")

    print(f"\nTotal folds created: {fold_num}")

    print("\nGenerating and saving split visualization...")
    fig, axes = plot_cv_splits(spatial_cv, X=X_spatial, y=y_spatial, coords=coords)

    if fig:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'spatial_splits.png')
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_spatial_example()