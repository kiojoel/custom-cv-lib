from abc import ABC, abstractmethod
import numpy as np

class BaseCVSplitter(ABC):
    """
    Base class for all custom cross-validators in this library.

    This class provides a consistent interface and enforces the implementation
    of the essential `split` method in all subclasses. It's designed to be
    compatible with the scikit-learn CV splitter API.
    """
    def __init__(self, n_splits):
        """
        Initializes the cross-validator.

        Args:
            n_splits (int): The number of splits or folds.
        """
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        self.n_splits = n_splits

    @abstractmethod
    def split(self, X, y=None, groups=None, **kwargs):
        """
        Abstract method to generate indices to split data into training and test set.

        Subclasses MUST implement this method.

        Args:
            X: Array-like of shape (n_samples, n_features). The input data.
            y: Array-like of shape (n_samples,). The target variable.
            groups: Array-like of shape (n_samples,). Group labels for the samples.
            **kwargs: Allows for additional parameters like 'coords' for spatial CV.

        Yields:
            train (np.array): The training set indices for that split.
            test (np.array): The testing set indices for that split.
        """
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        This method is provided by the base class and works for all subclasses
        that set `self.n_splits` in their `__init__`.
        """
        return self.n_splits

    def __repr__(self):
        """
        Provides a developer-friendly string representation of the object.
        """
        return f'{self.__class__.__name__}(n_splits={self.n_splits})'