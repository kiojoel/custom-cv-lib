import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from custom_cv.splitters import StratifiedKFoldCV

print("Testing our Custom StratifiedKFoldCV")

# 1. Create imbalanced dummy data
X, y = make_classification(
    n_samples=150, n_features=5, n_informative=3, n_redundant=0,
    n_classes=3, n_clusters_per_class=1,
    weights=[0.8, 0.1, 0.1], # 80% class 0, 10% class 1, 10% class 2
    random_state=42
)
print(f"Overall Class Distribution: {np.bincount(y)}")

# 2. Create an instance of our cross-validator
custom_cv = StratifiedKFoldCV(n_splits=5, shuffle=True, random_state=42)

# 3. Use the splitter
model = LogisticRegression()
fold_scores = []

for fold_num, (train_indices, test_indices) in enumerate(custom_cv.split(X, y)):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"\nFold {fold_num+1}:")
    # Check the class distribution in the test set of this fold
    print(f"  Test set class distribution: {np.bincount(y_test)}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    fold_scores.append(score)

# 4. Print final results
print("\n Final Results")
print(f"Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")