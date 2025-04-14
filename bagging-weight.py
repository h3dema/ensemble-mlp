import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from data import generate_data


if __name__ == "__main__":
    n_estimators_bagging = 10
    n_epochs = 50
    n_classes = 30

    # Generate a synthetic dataset
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=200,
        centers=n_classes,
        random_state=42,
        verbose=True,
        )

    # --- "Weighted Bagging-like" with Decision Trees ---
    print("\n--- 'Weighted Bagging-like' with Decision Trees ---")

    estimators_bagging = []
    sample_weights_bagging = np.ones(len(X_train)) / len(X_train)

    for i in range(n_estimators_bagging):
        print(f"\nBagging Round {i+1}...")

        # Resample training data based on current sample weights
        n_samples = len(X_train)
        probabilities = sample_weights_bagging
        indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probabilities)
        X_resampled, y_resampled = X_train[indices], y_train[indices]

        # Train a Decision Tree on the resampled data
        dt = DecisionTreeClassifier(max_depth=3, random_state=42 + i)
        dt.fit(X_resampled, y_resampled)
        estimators_bagging.append(dt)

        # Make predictions on the original training data
        y_train_pred = dt.predict(X_train)
        incorrect = (y_train_pred != y_train)
        error = np.sum(sample_weights_bagging[incorrect])

        print(f"  Weighted error: {error:.4f}")

        # Update sample weights (making it more like boosting)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-7)) # Small epsilon to avoid division by zero
        for j in range(len(X_train)):
            if incorrect[j]:
                sample_weights_bagging[j] *= np.exp(alpha)
            else:
                sample_weights_bagging[j] *= np.exp(-alpha)
        sample_weights_bagging /= np.sum(sample_weights_bagging)

    # Make predictions using majority voting
    predictions = np.array([est.predict(X_test) for est in estimators_bagging])
    y_pred_weighted_bagging = mode(predictions, axis=0)[0].flatten()

    # Evaluate the accuracy
    accuracy_weighted_bagging = accuracy_score(y_test, y_pred_weighted_bagging)
    print(f"\nAccuracy of 'Weighted Bagging-like': {accuracy_weighted_bagging:.4f}")