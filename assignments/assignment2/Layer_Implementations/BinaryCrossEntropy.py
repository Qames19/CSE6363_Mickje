########################################################################
# Student:      James Mick                                             #
# ID:           2234431                                                #
# Course:       CSE-6363                                               #
# Assignment:   2                                                      #
# Professor:    Dr. Alex Dilhoff                                       #
# Assisted by:  ChatGPT 4o                                             #
########################################################################
import numpy as np
from Layer_Implementations.Layer import Layer

class BinaryCrossEntropy(Layer):
    def __init__(self):
        """
        Initializes the ReLU activation layer        
        """
        pass

    def forward(self, y_true, y_pred):
        """
        Computes the Binary Cross-Entropy loss.

        Parameters:
        y_true (numpy.ndarray): True labels (0 or 1).
        y_pred (numpy.ndarray): Predicted probabilities (Sigmoid output).

        Returns:
        float: The Binary Cross-Entropy loss value.
        """
        # Compute and return ReLU forward pass ReLU(input) = max(0, input)
        self.inputs = y_true

        # initialize a small value, epsilon
        epsilon = 1e-9

        # Clip y predictions using epsilon to prevent taking log of either 0 or 1
        self.y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the Binary Cross-Entropy Loss
        bcel = -np.mean(self.inputs * np.log(self.y_pred) + (1 - self.inputs) * np.log(1 - self.y_pred))

        # Return the Binary Cross-Entropy Loss
        return bcel
        
    def backward(self):
        """
        Computes the gradient of the Binary Cross-Entropy Loss.

        Parameters:
        y_pred (numpy.ndarray): Predicted probabilities (Sigmoid output) from the next layer (batch_size, input_size).

        Returns:
        numpy.ndarray: Gradient with respect to inputs (batch_size, input_size).
        
        """
        # Compute the gradient of the Loss:  dL/dy_pred = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        bcel_grad = -(self.inputs / self.y_pred) + ((1 - self.inputs) / (1 - self.y_pred))

        # Return the grad_input
        return bcel_grad