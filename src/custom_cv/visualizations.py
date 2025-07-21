import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .splitters import TimeSeriesCV, SpatialBlockCV

def plot_cv_scores(results_dict, title='Cross-Validation Score Comparison'):
    """
    Creates a box plot of cross-validation scores for one or more models.

    This function creates and returns the matplotlib figure and axes for further
    customization, display, or saving.

    Args:
        results_dict (dict): A dictionary of model names and score lists.
        title (str): The title for the plot.

    Returns:
        tuple: A tuple containing the matplotlib (figure, axes) objects.
    """
    try:
        results_df = pd.DataFrame(results_dict)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        return None, None

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=results_df, palette='viridis', ax=ax)
    sns.stripplot(data=results_df, jitter=True, color="0.3", size=5, ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    for i, model_name in enumerate(results_df.columns):
        mean_score = np.mean(results_df[model_name])
        ax.text(i, mean_score, f'{mean_score:.3f}',
                ha='center', va='bottom', color='white',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    ax.grid(True, which='major', linestyle='--', linewidth='0.5')
    fig.tight_layout()

    return fig, ax


def plot_cv_splits(cv_splitter, X, y=None, coords=None):
    """
    Visualizes the train/test splits for a given CV strategy.

    Returns the matplotlib figure and axes for customization, display, or saving.

    Args:
        cv_splitter: An instantiated cross-validator object from your library.
        X (np.array): The feature data.
        y (np.array): The target data (optional).
        coords (np.array): The spatial coordinates, required for SpatialBlockCV.

    Returns:
        tuple: A tuple containing the matplotlib (figure, axes) objects.
    """
    if isinstance(cv_splitter, SpatialBlockCV):
        if coords is None:
            raise ValueError("For SpatialBlockCV, 'coords' must be provided.")

        n_splits = cv_splitter.n_splits
        n_cols = int(np.ceil(np.sqrt(n_splits)))
        n_rows = int(np.ceil(n_splits / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        axes = axes.flatten()

        splits = list(cv_splitter.split(X, y=y, coords=coords))

        for i, (train_idx, test_idx) in enumerate(splits):
            ax = axes[i]
            ax.scatter(coords[train_idx, 0], coords[train_idx, 1],
                       c='lightblue', marker='.', s=20, label='Train')
            if len(test_idx) > 0:
                ax.scatter(coords[test_idx, 0], coords[test_idx, 1],
                           c='red', marker='.', s=40, label='Test')
            ax.set_title(f"Fold {i+1}")
            ax.set_aspect('equal', adjustable='box')

        for i in range(len(splits), len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'SpatialBlockCV Splits ({cv_splitter.n_blocks_per_dim}x{cv_splitter.n_blocks_per_dim} Grid)', fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, axes

    # Handle General (Time-Series like) Visualization
    n_splits = cv_splitter.get_n_splits()
    fig, ax = plt.subplots(figsize=(12, n_splits * 0.5))
    cmap = plt.cm.viridis

    for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        ax.axhspan(i - 0.4, i + 0.4, color='whitesmoke', zorder=1)
        ax.scatter(train_idx, [i] * len(train_idx),
                   color=cmap(0.2), marker='|', s=50, lw=2, label='Train' if i == 0 else "", zorder=2)
        ax.scatter(test_idx, [i] * len(test_idx),
                   color=cmap(0.8), marker='|', s=50, lw=2, label='Test' if i == 0 else "", zorder=2)

    ax.set(yticks=np.arange(n_splits), yticklabels=[f"Fold {i+1}" for i in range(n_splits)],
           xlabel='Sample Index', ylabel="CV Fold",
           ylim=[n_splits - 0.5, -0.5], xlim=[-2, len(X) + 1])
    ax.set_title(f'{type(cv_splitter).__name__} Splits Visualization')
    ax.legend(loc='upper right')
    fig.tight_layout()

    return fig, ax