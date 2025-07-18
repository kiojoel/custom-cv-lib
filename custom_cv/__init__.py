"""
A custom cross-validation library for machine learning.

This library provides implementations for specialized cross-validation strategies
including stratified, time-series, and spatial methods, along with tools for
statistical significance testing and visualization of CV results.
"""

# Import the base class so it's available for extension, but not typically for end-users
from .base import BaseCVSplitter

# Import the specific splitter classes to make them directly accessible
from .splitters import (
    StratifiedKFoldCV,
    TimeSeriesCV,
    SpatialBlockCV
)

# Import the statistical testing functions
from .stats import paired_ttest_cv

# Import the visualization tools
from .visualizations import (
    plot_cv_scores,
    plot_cv_splits
)


__all__ = [
    'BaseCVSplitter',
    'StratifiedKFoldCV',
    'TimeSeriesCV',
    'SpatialBlockCV',
    'paired_ttest_cv',
    'plot_cv_scores',
    'plot_cv_splits'
]

__version__ = "0.1.0"