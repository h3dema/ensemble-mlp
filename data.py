import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_data(n_samples=200, centers=3, random_state=42, test_size=0.3, verbose=False):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        print("Unique classes in y_train:", np.unique(y_train).shape[0])

    return X_train, X_test, y_train, y_test
