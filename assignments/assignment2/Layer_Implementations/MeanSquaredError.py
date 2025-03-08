########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for Binary Cross-Entropy Loss.             #
########################################################################
import numpy as np
from Layer_Implementations.Layer import Layer
import matplotlib.pyplot as plt

class MSELoss(Layer):
    def __init__(self):
        """Initializes the Mean Squared Error Loss."""
        pass

    def forward(self, y_true, y_pred):
        """Computes the Mean Squared Error Loss."""
        # Ensure y_true is numpy array
        self.y_true = np.asarray(y_true)

        # Clip predictions before inverse transforming to prevent extreme values
        y_pred = np.clip(y_pred, -1e6, 1e6)

        # Ensure y_pred is numpy array and reshape if necessary
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Debug: Print y_pred statistics
        print(f"y_pred stats before expm1: mean={np.mean(y_pred)}, min={np.min(y_pred)}, max={np.max(y_pred)}")
        
        # Reverse log transform and scaler applied to y_train and y_test
        try:
            self.y_pred = np.expm1(y_pred)
        except AttributeError as e:
            print(f"Error: {e}")
            print(f"Type of y_pred: {type(y_pred)}")
            print(f"Content of y_pred: {y_pred}")
            raise

        # Debug: Print y_pred statistics after expm1
        print(f"y_pred stats after expm1: mean={np.mean(self.y_pred)}, min={np.min(self.y_pred)}, max={np.max(self.y_pred)}")

        # Compute the MSE loss
        return np.mean(np.square(self.y_true - self.y_pred))

    def backward(self):
        """Computes the gradient of MSE Loss."""
        return -2 * (self.y_true - self.y_pred) / len(self.y_true)

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, loss_function):
    """
    Trains the neural network model with early stopping and loss tracking.

    Parameters:
    - model: Sequential model instance
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - epochs: Maximum number of epochs
    - batch_size: Size of each training batch
    - lr: Learning rate for gradient descent
    - loss_function: Loss function to optimize

    Returns:
    - training_losses, validation_losses: Lists of loss values per epoch
    """
    training_losses = []
    validation_losses = []
    best_val_loss = float("inf")
    patience = 3  # Stop training after 3 epochs of no improvement
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            predictions = model.forward(X_batch)

            # Ensure predictions are numpy arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.asarray(predictions)

            # Compute loss
            loss = loss_function.forward(y_batch, predictions)

            # Check for invalid values - ChatGPT suggested fix for runtime errors.
            if np.isnan(loss) or np.isinf(loss):
                print(f"Detected NaN or Inf loss at epoch {epoch}. Stopping training.")
                return
            
            # Update loss
            total_loss += loss

            # Backward pass
            grad_loss = loss_function.backward()
            model.backward(grad_loss)

            # Set lambda_reg - ChatGPT suggestion
            lambda_reg = 0.01
            # Update weights
            for layer in model.layers:
                if hasattr(layer, "weights") and hasattr(layer, "bias"):
                    np.clip(layer.grad_weights, -1, 1, out=layer.grad_weights)
                    np.clip(layer.grad_bias, -1, 1, out=layer.grad_bias)

                    layer.weights -= lr * (layer.grad_weights + lambda_reg * layer.weights)
                    layer.bias -= lr * layer.grad_bias

        # Compute validation loss
        val_predictions = model.forward(X_val)
        
        # Ensure val_predictions are numpy arrays
        if not isinstance(val_predictions, np.ndarray):
            val_predictions = np.asarray(val_predictions)
        
        val_loss = loss_function.forward(y_val, val_predictions)

        training_losses.append(total_loss / len(X_train))
        validation_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {training_losses[-1]:.6f}, Validation Loss: {val_loss:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered! Stopping training.")
            break

    # Plot loss curves
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    return training_losses, validation_losses
