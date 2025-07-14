import numpy as np
from collections import defaultdict


class StratifiedKFoldCV:
  """
  Custom Stratified K-Fold cross-validator.
  Ensures each fold has a similar class distribution
  to the whole dataset.
  """

  def __init__(self, n_splits=5, shuffle=True, random_state=None):
    if n_splits <= 1:
      raise ValueError("n_splits must be greater than 1")
    self.n_splits = n_splits
    self.shuffle = shuffle
    self.random_state = random_state

  def split(self, X, y):
      """Generates indices to split data into training and test set."""
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

  def get_n_splits(self, X=None, y=None, groups=None):
      """Returns the number of splitting iterations."""
      return self.n_splits