from scipy import stats
import numpy as np

def paired_ttest_cv(scores_model1, scores_model2, alpha=0.05):
    """
    Performs a paired t-test on the cross-validation scores of two models
    to determine if the difference in their performance is statistically significant.

    The null hypothesis is that the true mean difference between the paired
    samples is zero.

    Args:
        scores_model1 (list or np.array): List of scores for model 1 for each CV fold.
        scores_model2 (list or np.array): List of scores for model 2 for each CV fold.
                                          Must be from the exact same folds.
        alpha (float): The significance level, typically 0.05.

    Returns:
        tuple: A tuple containing:
               - t_statistic (float): The calculated t-statistic.
               - p_value (float): The calculated p-value.
               - is_significant (bool): True if p_value < alpha, False otherwise.
    """
    scores_model1 = np.asarray(scores_model1)
    scores_model2 = np.asarray(scores_model2)

    if scores_model1.shape != scores_model2.shape:
        raise ValueError("Score lists for both models must have the same shape.")


    t_statistic, p_value = stats.ttest_rel(scores_model1, scores_model2)

    is_significant = p_value < alpha

    return t_statistic, p_value, is_significant