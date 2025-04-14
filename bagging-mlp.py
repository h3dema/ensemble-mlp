"""
This code performs a "weighted bagging-like" process with MLPs.
It trains multiple MLPs on resampled data (where the resampling is
influenced by sample weights). Then, for increasing thresholds of the
MLP's hidden layer size, it calculates weights for the MLPs
that meet the size criteria based on their performance on the weighted training data.
Finally, it combines the predictions of these selected and weighted MLPs
using a weighted average of their probabilities.

This approach allows you to see how the performance of an ensemble changes
as you include MLPs with larger hidden layer sizes and how their weights
are determined based on their individual contributions to classifying
the weighted training data. Remember that this is still a deviation from
standard bagging due to the weighted resampling and the explicit calculation and
application of estimator weights.
"""


import random
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from data import generate_data

# Hide convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

    all_estimators = []
    all_hidden_sizes = []
    sample_weights_history = [np.ones(len(X_train)) / len(X_train)]

    # Train the initial set of estimators
    for i in range(n_estimators_bagging):
        hidden_layer_size = random.randint(6, 12)
        all_hidden_sizes.append(hidden_layer_size)

        # Resample training data based on current sample weights
        n_samples = len(X_train)
        probabilities = sample_weights_history[-1]
        indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probabilities)
        X_resampled, y_resampled = X_train[indices], y_train[indices]

        mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation='relu',
                            max_iter=n_epochs, random_state=42 + i, learning_rate_init=0.1)
        mlp.fit(X_resampled, y_resampled)
        all_estimators.append(mlp)

    # Evaluate performance for increasing size thresholds
    for size_threshold in range(6, 13):
        print(f"\n--- Considering estimators with hidden_layer_size <= {size_threshold} ---")
        estimators_subset = [est for i, est in enumerate(all_estimators) if all_hidden_sizes[i] <= size_threshold]
        hidden_sizes_subset = [size for size in all_hidden_sizes if size <= size_threshold]

        if not estimators_subset:
            print("  No estimators meet this size criteria.")
            continue

        estimator_weights = []
        current_sample_weights = sample_weights_history[-1].copy()

        for idx, estimator in enumerate(estimators_subset):
            print(f"\n  Processing estimator {idx + 1} (size: {hidden_sizes_subset[idx]})")
            y_train_pred = estimator.predict(X_train)
            incorrect = (y_train_pred != y_train)
            weighted_error = np.sum(current_sample_weights[incorrect])
            print(f"    Weighted error: {weighted_error:.4f}")

            if weighted_error >= 0.5 or weighted_error <= 1e-7:
                estimator_weight = 1.0
            else:
                estimator_weight = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-7))

            estimator_weights.append(estimator_weight)
            print(f"    Estimator weight: {estimator_weight:.4f}")

            # Update sample weights for the next estimator in this subset
            for j in range(len(X_train)):
                if incorrect[j]:
                    current_sample_weights[j] *= np.exp(estimator_weight)
                else:
                    current_sample_weights[j] *= np.exp(-estimator_weight)
            current_sample_weights /= np.sum(current_sample_weights)

        sample_weights_history.append(current_sample_weights)

        # Make predictions using weighted majority voting
        final_predictions = np.zeros((len(X_test), n_classes))
        for i, estimator in enumerate(estimators_subset):
            probs = estimator.predict_proba(X_test)
            if probs.shape[1] != n_classes:
                # due to resampling, some classes may not be present in the test set
                # replace missing classes with zero probabilities
                probs = np.stack([np.zeros_like(probs[:, 0]) if c not in estimator.classes_ else probs[:, np.searchsorted(estimator.classes_, c).item()] for c in range(n_classes)], axis=1)

            final_predictions += estimator_weights[i] * probs

        y_pred_weighted_bagging = np.argmax(final_predictions, axis=1)
        accuracy_weighted_bagging = accuracy_score(y_test, y_pred_weighted_bagging)
        print(f"\n  Accuracy with hidden_layer_size <= {size_threshold}: {accuracy_weighted_bagging:.4f}")
