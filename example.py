import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from custom_cv.splitters import StratifiedKFoldCV, TimeSeriesCV

print("Testing our Custom StratifiedKFoldCV")

# Create imbalanced dummy data
X, y = make_classification(
    n_samples=150, n_features=5, n_informative=3, n_redundant=0,
    n_classes=3, n_clusters_per_class=1,
    weights=[0.8, 0.1, 0.1], # 80% class 0, 10% class 1, 10% class 2
    random_state=42
)
print(f"Overall Class Distribution: {np.bincount(y)}")

#  Create an instance of our cross-validator
custom_cv = StratifiedKFoldCV(n_splits=5, shuffle=True, random_state=42)

# Use the splitter
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

# Print final results
print("\n Final Results")
print(f"Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")



print("Testing our Custom TimeSeriesCV")

n_samples = 50
X_time = np.arange(n_samples).reshape(-1, 1)

y_time = X_time.flatten() * 2 + np.random.rand(n_samples) * 5

ts_cv = TimeSeriesCV(n_splits=5,test_size=3,gap=2)

print(f"Splitting {n_samples} samples into {ts_cv.get_n_splits()} folds:")
model = LinearRegression()
fold_scores = []

for fold_num, (train_indices, test_indices) in enumerate(ts_cv.split(X_time)):
    print(f"\nFold {fold_num+1}:")
    print(f" Train indices: {train_indices}")
    print(f" Test indices: {test_indices}")

    if train_indices.size > 0:
        print(f"  --> Max train index ({train_indices.max()}) is less than min test index ({test_indices.min()})")

    # Train the model and get score
    X_train, X_test = X_time[train_indices], X_time[test_indices]
    y_train, y_test = y_time[train_indices], y_time[test_indices]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # For regression, we often use Mean Squared Error (lower is better)
    score = mean_squared_error(y_test, predictions)
    fold_scores.append(score)

# Print final results
print("\n--- Final Time-Series Results ---")
print(f"Mean Squared Error for each fold: {[f'{s:.2f}' for s in fold_scores]}")
print(f"Average MSE: {np.mean(fold_scores):.2f}")