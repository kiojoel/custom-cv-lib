import numpy as np
from collections import defaultdict
from .base import BaseCVSplitter

class StratifiedKFoldCV(BaseCVSplitter):
    """
    Our custom Stratified K-Fold cross-validator.
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=None):

        super().__init__(n_splits=n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None, **kwargs):
        y = np.asarray(y)
        n_samples = len(y)
        class_indices = defaultdict(list)
        for i, class_label in enumerate(y):
            class_indices[class_label].append(i)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            for indices in class_indices.values():
                rng.shuffle(indices)
        class_folds = defaultdict(list)
        for class_label, indices in class_indices.items():
            fold_sizes = np.full(self.n_splits, len(indices) // self.n_splits, dtype=int)
            fold_sizes[:len(indices) % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                class_folds[class_label].append(indices[start:stop])
                current = stop
        all_indices = np.arange(n_samples)
        for i in range(self.n_splits):
            test_indices = np.concatenate([folds[i] for folds in class_folds.values()])
            train_indices = np.setdiff1d(all_indices, test_indices)
            yield train_indices, test_indices



class TimeSeriesCV(BaseCVSplitter):
    """
    Our custom Time-Series cross-validator.
    """
    def __init__(self, n_splits=5, test_size=1, gap=0):
        super().__init__(n_splits=n_splits)
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None, **kwargs):
        n_samples = len(X)
        indices = np.arange(n_samples)
        total_test_plus_gaps = (self.n_splits * self.test_size) + ((self.n_splits - 1) * self.gap)
        first_test_start = n_samples - total_test_plus_gaps
        if first_test_start <= 0:
            raise ValueError(f"Cannot create {self.n_splits} folds with test_size={self.test_size} and gap={self.gap}.")
        for i in range(self.n_splits):
            test_start = first_test_start + i * (self.test_size + self.gap)
            test_end = test_start + self.test_size
            train_end = test_start - self.gap
            train_indices = indices[0:train_end]
            test_indices = indices[test_start:test_end]
            if len(train_indices) == 0:
                raise ValueError("The first training set has 0 samples.")
            yield train_indices, test_indices



class SpatialBlockCV(BaseCVSplitter):
    """
    Our custom Spatial Block cross-validator.
    """
    def __init__(self, n_blocks_per_dim=4):
        n_splits = n_blocks_per_dim ** 2
        super().__init__(n_splits=n_splits)
        self.n_blocks_per_dim = n_blocks_per_dim

    def split(self, X, y=None, groups=None, **kwargs):
        coords = kwargs.get('coords')
        if coords is None:
            raise ValueError("SpatialBlockCV requires a 'coords' array passed as a keyword argument.")
        coords = np.asarray(coords)
        if coords.shape[0] != X.shape[0]:
            raise ValueError("X and coords must have the same number of samples.")
        if coords.shape[1] != 2:
            raise ValueError("coords must have 2 columns (for x and y).")
        n_samples = X.shape[0]
        all_indices = np.arange(n_samples)
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        x_bins = np.linspace(x_coords.min(), x_coords.max(), self.n_blocks_per_dim + 1)
        y_bins = np.linspace(y_coords.min(), y_coords.max(), self.n_blocks_per_dim + 1)
        x_block_assignment = np.digitize(x_coords, x_bins)
        y_block_assignment = np.digitize(y_coords, y_bins)
        x_block_assignment[x_block_assignment > self.n_blocks_per_dim] = self.n_blocks_per_dim
        y_block_assignment[y_block_assignment > self.n_blocks_per_dim] = self.n_blocks_per_dim
        block_ids = (y_block_assignment - 1) * self.n_blocks_per_dim + (x_block_assignment - 1)
        unique_block_ids = np.unique(block_ids)
        for block_id in unique_block_ids:
            test_indices = all_indices[block_ids == block_id]
            train_indices = all_indices[block_ids != block_id]
            if len(test_indices) > 0 and len(train_indices) > 0:
                yield train_indices, test_indices