import numpy as np
import pytest
from custom_cv import StratifiedKFoldCV, TimeSeriesCV, SpatialBlockCV

# --- Tests for StratifiedKFoldCV

def test_stratified_kfold_n_splits():
    """Verify that the splitter yields the correct number of splits."""
    X = np.zeros((10, 2))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cv = StratifiedKFoldCV(n_splits=5)
    splits = list(cv.split(X, y))
    assert len(splits) == 5

def test_stratified_kfold_maintains_ratio():
    """Verify that each fold maintains the class ratio."""
    # 80 samples of class 0, 20 samples of class 1 (4:1 ratio)
    y = np.array([0] * 80 + [1] * 20)
    X = np.zeros((100, 2))
    cv = StratifiedKFoldCV(n_splits=10, shuffle=False) # No shuffle for deterministic test

    for train_idx, test_idx in cv.split(X, y):
        y_test = y[test_idx]
        # Each test set should have 10 samples
        assert len(y_test) == 10
        # It should have 8 samples of class 0 and 2 of class 1
        assert np.sum(y_test == 0) == 8
        assert np.sum(y_test == 1) == 2

# --- Tests for TimeSeriesCV ---

def test_timeseries_no_overlap_and_gap():
    """Verify that train indices are always before test indices and respect the gap."""
    X = np.zeros((20, 2))
    cv = TimeSeriesCV(n_splits=5, test_size=2, gap=1)

    for train_idx, test_idx in cv.split(X):
        # The maximum training index must be less than the minimum test index
        assert train_idx.max() < test_idx.min()
        # The gap should be respected
        assert test_idx.min() - train_idx.max() > 1

def test_timeseries_raises_error_on_small_data():
    """Verify that it raises a ValueError if the data is too small for the splits."""
    X = np.zeros((10, 2))
    cv = TimeSeriesCV(n_splits=5, test_size=2, gap=1)
    # This should fail, so we check that it raises the correct error
    with pytest.raises(ValueError, match="Cannot create 5 folds"):
        list(cv.split(X))

# --- Tests for SpatialBlockCV ---

def test_spatial_no_shared_points():
    """Verify that train and test sets have no common points."""
    X = np.random.rand(100, 2)
    coords = np.random.rand(100, 2)
    cv = SpatialBlockCV(n_blocks_per_dim=4)

    for train_idx, test_idx in cv.split(X, coords=coords):
        # The intersection of the two sets of indices should be empty
        assert len(np.intersect1d(train_idx, test_idx)) == 0

def test_spatial_all_points_covered():
    """Verify that across all test folds, every point is used exactly once."""
    X = np.random.rand(100, 2)
    coords = np.random.rand(100, 2)
    cv = SpatialBlockCV(n_blocks_per_dim=5)

    all_test_indices = []
    for train_idx, test_idx in cv.split(X, y=None, coords=coords):
        all_test_indices.extend(test_idx)

    # The number of unique test indices should equal the total number of samples
    assert len(np.unique(all_test_indices)) == len(X)