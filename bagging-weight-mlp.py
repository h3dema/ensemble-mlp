"""
The "weighted" aspect here refers to the potential to use non-uniform `initial_sample_weights`
if desired. Standard Bagging in sklearn uses uniform random sampling by default,
which is equivalent to having equal initial weights.
If you wanted to bias the bootstrap samples from the beginning,
you could adjust initial_sample_weights.

This version aligns more closely (compared to bagging-mlp) with the standard bagging paradigm:
train multiple independent learners on bootstrap samples of the data and
combine their predictions without an adaptive feedback loop that
updates sample weights based on performance.
"""

import warnings
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.exceptions import ConvergenceWarning

# Hide convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data import generate_data


if __name__ == "__main__":
    n_estimators_bagging = 20
    n_epochs = 50
    n_classes = 30
    random_state=42

    # Generate a synthetic dataset
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=200,
        centers=n_classes,
        random_state=random_state,
        verbose=False,
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
        hidden_layer_size = random.randint(6, 12)
        mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            activation='relu',
            max_iter=n_epochs,
            random_state=random_state + i,
            learning_rate_init=0.1,
        )
        mlp.fit(X_resampled, y_resampled)
        estimators_bagging.append(mlp)

        # Make predictions on the original training data
        y_train_pred = mlp.predict(X_train)
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
    # use the mode (most frequent prediction) as the final prediction
    y_pred_weighted_bagging = mode(predictions, axis=0)[0].flatten()

    # Evaluate the accuracy
    accuracy_weighted_bagging = accuracy_score(y_test, y_pred_weighted_bagging)
    print(f"\nAccuracy of 'Weighted Bagging-like': {accuracy_weighted_bagging:.4f}")

    # --- Alternative: Averaging Probabilities ---
    print("\n--- Alternative: 'Weighted Bagging-like' with MLPs (Averaging Probabilities) ---")

    final_probabilities = np.zeros((len(X_test), n_classes))
    for estimator in estimators_bagging:
        probs = estimator.predict_proba(X_test)
        if probs.shape[1] != n_classes:
            # due to resampling, some classes may not be present in the test set
            # replace missing classes with zero probabilities
            probs = np.stack([np.zeros_like(probs[:, 0]) if c not in estimator.classes_ else probs[:, np.searchsorted(estimator.classes_, c).item()] for c in range(n_classes)], axis=1)

        final_probabilities += probs
    final_probabilities /= n_estimators_bagging
    y_pred_avg_probs = np.argmax(final_probabilities, axis=1)
    accuracy_avg_probs = accuracy_score(y_test, y_pred_avg_probs)
    print(f"Accuracy of 'Weighted Bagging-like' with MLPs (Averaging Probabilities): {accuracy_avg_probs:.4f}")
