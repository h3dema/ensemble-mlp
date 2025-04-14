"""
This code implements a form of Gradient Boosting using MLPs
as the base learners by sequentially training MLPs to predict the residuals
of the ensemble's predictions.

However, it's important to consider the trade-offs in terms of complexity,
computational cost, and interpretability compared to
traditional tree-based Gradient Boosting (see sklearn.ensemble.GradientBoostingClassifier).

"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data import generate_data

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax along the class dimension

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).numpy()


if __name__ == "__main__":
    n_estimators_gbm_mlp = 5  # Number of boosting rounds (MLPs)
    random_state_base = 42
    n_epochs = 200
    n_classes = 30
    hidden_size = 10
    batch_size = 32

    boosting_learning_rate = 0.1  # Learning rate for the boosting process
    use_decay = True  # Whether to use learning rate decay for MLPs
    if use_decay:
        """
            Integrating learning rate decay into the Gradient Boosting with MLPs involves adjusting
            the learning rate of the optimizer used to train each individual MLP in the boosting rounds.

            **Note**: This LR values is diffrent from the `boosting_learning_rate`
        """
        initial_mlp_lr = 0.01      # Initial learning rate for each MLP
        lr_decay_factor = 0.9      # Factor by which the MLP learning rate decays
        lr_decay_frequency = 1     # Decay the LR every this many boosting rounds


    # Generate a synthetic dataset
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=200,
        centers=n_classes,
        random_state=random_state_base,
        verbose=True,
        )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Convert labels to one-hot encoding for multi-class classification with MLP in boosting
    label_binarizer = LabelBinarizer()

    # **NOTE**: what if y_train does not contain all classes?
    y_train_one_hot = label_binarizer.fit_transform(y_train)
    y_train_one_hot_tensor = torch.tensor(y_train_one_hot, dtype=torch.float32)

    # --- Gradient Boosting with MLPs ---
    print("\n--- Gradient Boosting with MLPs ---")

    ensemble_predictions = torch.zeros((len(X_train), n_classes))  # Accumulate predictions

    input_size = X_train.shape[1]
    estimators_gbm_mlp = []

    criterion = nn.MSELoss()  # Using MSE for residual fitting
    current_mlp_lr = initial_mlp_lr

    for i in range(n_estimators_gbm_mlp):
        print(f"\nMLP Boosting Round {i+1}...")

        # Calculate the residual (the error from the current ensemble)
        residuals = y_train_one_hot_tensor - ensemble_predictions.detach()

        # Train a new MLP to predict the residuals
        mlp = MLP(input_size, hidden_size, n_classes)
        optimizer_params = {'lr': current_mlp_lr}
        optimizer = optim.Adam(mlp.parameters(), **optimizer_params)
        dataloader = DataLoader(
            TensorDataset(
                X_train_tensor,
                residuals
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        mlp.train()
        for epoch in range(n_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = mlp(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        """
        Gradient Boosting builds the final strong predictor in a stage-wise fashion.
        It adds weak learners (in this case, MLPs) to the ensemble sequentially.
        The prediction of the ensemble at any stage is the sum of the predictions of
        all the weak learners trained so far.

        the learning_rate acts as a form of L2-like regularization on the ensemble's weights
        """
        mlp.eval()
        with torch.no_grad():
            current_predictions = mlp(X_train_tensor)
            ensemble_predictions += boosting_learning_rate * current_predictions
            estimators_gbm_mlp.append(mlp)

        if use_decay:
            # Decay the learning rate of the MLP optimizer
            if (i + 1) % lr_decay_frequency == 0 and current_mlp_lr > 1e-6:  # Avoid going too low
                current_mlp_lr *= lr_decay_factor


    # Make predictions on the test set
    final_predictions_proba = np.zeros((len(X_test), n_classes))
    for estimator in estimators_gbm_mlp:
        final_predictions_proba += boosting_learning_rate * estimator.predict_proba(X_test_tensor)

    # Convert probabilities to class labels
    y_pred_proba_np = final_predictions_proba.numpy() if isinstance(final_predictions_proba, torch.Tensor) else final_predictions_proba
    y_pred_gbm_mlp = label_binarizer.inverse_transform(y_pred_proba_np)

    # Evaluate the accuracy
    accuracy_gbm_mlp = accuracy_score(y_test, y_pred_gbm_mlp)
    print(f"\nAccuracy of Gradient Boosting with MLPs: {accuracy_gbm_mlp:.4f}")
