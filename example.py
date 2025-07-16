import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier


from custom_cv.splitters import StratifiedKFoldCV, TimeSeriesCV, SpatialBlockCV
from custom_cv.visualizations import plot_cv_scores

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


print("\n\n--- Testing our Custom SpatialBlockCV ---")

# Dummy spatial data
np.random.seed(42)
n_samples = 200
X_spatial = np.random.rand(n_samples, 5) # 5 random features

coords1 = np.random.rand(n_samples // 2, 2) * 50  # Cluster in bottom-left
coords2 = (np.random.rand(n_samples // 2, 2) * 50) + 50 # Cluster in top-right
coords = np.vstack([coords1, coords2])

# A simple target variable related to the coordinates
y_spatial = coords[:, 0] + coords[:, 1] + np.random.randn(n_samples) * 5

# Create an instance of our spatial cross-validator
# Let's create a 4x4 grid, resulting in 16 potential folds.
spatial_cv = SpatialBlockCV(n_blocks_per_dim=4)

print(f"Splitting {n_samples} spatial samples into a {spatial_cv.n_blocks_per_dim}x{spatial_cv.n_blocks_per_dim} grid.")

#  Use the splitter to see the indices
fold_num = 0
for train_indices, test_indices in spatial_cv.split(X=X_spatial, coords=coords):
    fold_num += 1
    # Get the coordinates for this fold's test set
    test_coords = coords[test_indices]

    # Get the bounding box of the test coordinates
    min_x, min_y = test_coords.min(axis=0)
    max_x, max_y = test_coords.max(axis=0)

    print(f"\nFold {fold_num}:")
    print(f"  Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print(f"  Test points are all within a bounding box:")
    print(f"    x range: [{min_x:.1f}, {max_x:.1f}]")
    print(f"    y range: [{min_y:.1f}, {max_y:.1f}]")

print(f"\nTotal folds created: {fold_num}")


print("--- Testing our Custom StratifiedKFoldCV with Visualization ---")

# Create dummy data
X, y = make_classification(
    n_samples=500, n_features=20, n_informative=5, n_redundant=2,
    n_classes=3, weights=[0.6, 0.3, 0.1], flip_y=0.05, random_state=42
)

# Define the models we want to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
}

# Define our custom cross-validator
custom_cv = StratifiedKFoldCV(n_splits=10, shuffle=True, random_state=42)

# Loop through each model and get its CV scores
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
    print(f"{model_name} - Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

print("\nGenerating visualization...")
plot_cv_scores(all_results, title='Stratified CV Performance: Logistic Regression vs. Random Forest')