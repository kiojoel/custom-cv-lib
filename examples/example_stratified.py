import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

from custom_cv import (
    StratifiedKFoldCV,
    plot_cv_scores,
    paired_ttest_cv
)

def run_stratified_example():
    print("--- Running StratifiedKFoldCV Example ---")

    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=5, n_redundant=2,
        n_classes=3, weights=[0.6, 0.3, 0.1], flip_y=0.05, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }

    custom_cv = StratifiedKFoldCV(n_splits=10, shuffle=True, random_state=42)

    all_results = {}
    for model_name, model in models.items():
        fold_scores = []
        for train_indices, test_indices in custom_cv.split(X, y):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            fold_scores.append(score)
        all_results[model_name] = fold_scores
        print(f"{model_name} - Mean Accuracy: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")

    print("\n--- Statistical Significance Test ---")
    lr_scores = all_results['Logistic Regression']
    rf_scores = all_results['Random Forest']
    t_stat, p_value, is_significant = paired_ttest_cv(rf_scores, lr_scores)

    print(f"Comparing Random Forest vs. Logistic Regression:")
    print(f"  T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if is_significant:
        winner = "Random Forest" if np.mean(rf_scores) > np.mean(lr_scores) else "Logistic Regression"
        print(f"  Conclusion: The performance of {winner} is significantly better.")
    else:
        print("  Conclusion: The difference in performance is not statistically significant.")

    print("\nGenerating and saving visualization...")
    fig, ax = plot_cv_scores(all_results, title='Stratified CV Performance')

    if fig:
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        save_path = os.path.join(plots_dir, 'stratified_cv_comparison.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_stratified_example()