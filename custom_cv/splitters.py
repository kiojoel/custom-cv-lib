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



class TimeSeriesCV:
   """
   Custom Time-Series cross-validator.
   Creates folds that respect the temporal order of the
   data.
   The training set always come befor the test set.
   """
   def __init__(self, n_splits = 5, test_size = 1, gap = 0):
      """
      Args:
      n_splits (int): The number of splits/folds.
      test_size (int): The number of samples in each test set.
      gap (int): The number of samples to skip between the end of the train set
      and the beginning of the test set. This prevents leakage from the data points
      that might be too cose in time.
      """
      if n_splits <= 1:
         raise ValueError("n_splits must be greater than 1.")

      self.n_splits = n_splits
      self.test_size = test_size
      self.gap = gap

   def split(self, X, y = None, groups = None):
      """
      Generates Indices to split data into training
      and test set
      """
      n_samples = len(X)
      indices = np.arange(n_samples)

      total_test_plus_gaps = (self.n_splits * self.test_size) + ((self.n_splits - 1) * self.gap)
      first_test_start = n_samples - total_test_plus_gaps

      if first_test_start <= 0:
         raise ValueError(f"Cannot create {self.n_splits} folds with test_size={self.test_size}"
                          f"and gap={self.gap}. The dataset of size {n_samples} is too small.")

      for i in range (self.n_splits):
         test_start = first_test_start + i * (self.test_size + self.gap)
         test_end = test_start + self.test_size
         train_end = test_start - self.gap
         train_indices = indices[0:train_end]
         test_indices = indices[test_start:test_end]

         if len(train_indices) == 0:
            raise ValueError("The first training set has 0 samples. adjust parameters.")

         yield train_indices,test_indices

   def get_n_splits(self, X=None, y=None, groups=None):
      """Returns the number of splitting iterations."""
      return self.n_splits