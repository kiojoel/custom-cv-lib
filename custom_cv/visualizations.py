import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_cv_scores(results_dict, title='Cross-Validation Score Comparison'):
    """
    Creates a box plot of cross-validation scores for one or more models.

    Args:
        results_dict (dict): A dictionary where keys are model names (str)
        and values are lists of scores (list of floats).
        title (str): The title for the plot.
    """

    try:
        results_df = pd.DataFrame(results_dict)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        print("Please ensure all models were evaluated on the same number of CV folds.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=results_df, palette='viridis')
    sns.stripplot(data=results_df, jitter=True, color=".3", size=5)


    plt.title(title, fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add a horizontal line for the average score of each model
    for i, model_name in enumerate(results_df.columns):
        mean_score = np.mean(results_df[model_name])
        plt.text(i, mean_score, f'{mean_score:.3f}',
                 ha='center', va='bottom', color='white',
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.grid(True, which='major', linestyle='--', linewidth='0.5')
    plt.tight_layout()

    plt.show()