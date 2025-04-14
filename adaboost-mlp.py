"""
AdaBoost-CNN is a type of ensemble learning method that combines the strengths of the AdaBoost (Adaptive Boosting) algorithm with Convolutional Neural Networks (CNNs). Here's a breakdown of how it generally works:  
In AdaBoost-CNN, the "weak learners" are typically CNN models (though they could be other types of classifiers that extract features from images)

--> ** This code is a simplified version of AdaBoost using MLPClassifier (Multi-layer Perceptron) from sklearn.**
It is not a direct implementation of AdaBoost-CNN, but rather an illustration of the AdaBoost concept with simple neural networks.
 The code uses MLPClassifier as the weak learner and applies the AdaBoost algorithm to combine their predictions.**

## Algorithm Fundamentals:

Ensemble Learning: AdaBoost is an ensemble learning technique. This means it combines multiple "weak learners" to create a single "strong learner" with improved accuracy.
Iterative Training: AdaBoost trains a sequence of weak classifiers iteratively.  
Weighted Samples: In each iteration, AdaBoost assigns weights to the training samples. Initially, all samples have equal weights.  
Focus on Misclassified Samples: After each weak learner is trained, the weights of the misclassified samples are increased. This forces subsequent weak learners to focus more on the examples that were difficult to classify correctly in the previous rounds.  
Weighted Weak Learners: Each trained weak learner is also assigned a weight based on its performance. More accurate learners get higher weights.
Final Prediction: The final prediction is made by combining the weighted predictions of all the weak learners (e.g., through a weighted majority vote for classification).

"""

import random
import warnings
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# Hide convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data import generate_data

if __name__ == "__main__":
    n_estimators_cnn = 10
    n_epochs = 50
    n_classes = 30

    # Generate a synthetic dataset
    X_train, X_test, y_train, y_test = generate_data(n_samples=200, centers=n_classes, random_state=42)

    # --- "AdaBoost-like" with Simple Neural Networks (Conditional Estimator Weights) ---
    print("\n--- 'AdaBoost-like' with Simple Neural Networks (Conditional Estimator Weights) ---")

    estimators = []
    estimator_weights_history = []  # Store weights for each size threshold
    sample_weights_history = [np.ones(len(X_train)) / len(X_train)] # Store sample weights

    # Train the initial set of estimators
    all_estimators = []
    all_hidden_sizes = []
    for i in range(n_estimators_cnn):
        hidden_layer_size = random.randint(6, 12)
        all_hidden_sizes.append(hidden_layer_size)

        # Resample training data based on current sample weights
        n_samples = len(X_train)
        probabilities = sample_weights_history[-1]
        indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probabilities)
        X_resampled, y_resampled = X_train[indices], y_train[indices]


        # Train with the latest sample weights
        mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            activation='relu',
            max_iter=n_epochs, random_state=42 + i,
            learning_rate_init=0.1
        )
        mlp.fit(X_resampled, y_resampled)
        all_estimators.append(mlp)

    # Compute estimator weights and evaluate performance for increasing size thresholds
    for size_threshold in range(6, 13):
        print(f"\n--- Considering estimators with hidden_layer_size <= {size_threshold} ---")
        estimators_subset = [est for i, est in enumerate(all_estimators) if all_hidden_sizes[i] <= size_threshold]
        hidden_sizes_subset = [size for size in all_hidden_sizes if size <= size_threshold]

        if not estimators_subset:
            print("  No estimators meet this size criteria.")
            estimator_weights_history.append([])
            continue

        current_estimator_weights = []
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
                estimator_weight = 0.5 * np.log((1 - weighted_error) / weighted_error)

            current_estimator_weights.append(estimator_weight)
            print(f"    Estimator weight: {estimator_weight:.4f}")

            # Update sample weights for the next estimator in this subset
            for j in range(len(X_train)):
                if incorrect[j]:
                    current_sample_weights[j] *= np.exp(estimator_weight)
                else:
                    current_sample_weights[j] *= np.exp(-estimator_weight)
            current_sample_weights /= np.sum(current_sample_weights)

        estimator_weights_history.append(current_estimator_weights)
        sample_weights_history.append(current_sample_weights)

        # Make predictions using the current subset of estimators
        final_predictions = np.zeros((len(X_test), n_classes))
        for i, estimator in enumerate(estimators_subset):
            probs = estimator.predict_proba(X_test)
            if probs.shape[1] != n_classes:
                # due to resampling, some classes may not be present in the test set
                # replace missing classes with zero probabilities
                probs = np.stack([np.zeros_like(probs[:, 0]) if c not in estimator.classes_ else probs[:, np.searchsorted(estimator.classes_, c).item()] for c in range(n_classes)], axis=1)

            final_predictions += current_estimator_weights[i] * probs

        y_pred_conditional = np.argmax(final_predictions, axis=1)
        accuracy_conditional = accuracy_score(y_test, y_pred_conditional)
        print(f"\n  Accuracy with hidden_layer_size <= {size_threshold}: {accuracy_conditional:.4f}")

    print("\nEstimator Weights History (for each size threshold):")
    for i, weights in enumerate(estimator_weights_history):
        print(f"Size <= {i + 6}: {weights}")