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

class Sigmoid(Layer):
    def __init__(self):
        """
        Initializes the Sigmoid activation layer.
        """
        pass

    def forward(self, inputs):
        """
        Performs a forward pass of the sigmoid layer

        Parameters:
        inputs (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output values in the range (0, 1).
        """
        # Store inputs to use for backpropagation 
        self.inputs = inputs

        # Calculate the sigmoid and store results
        self.outputs = 1 / (1 + np.exp(-self.inputs))

        # return the sigmoid
        return self.outputs
    
    def backward(self, grad_output):
        """
        Computes gradients for the backpropagation of sigmoid layer.

        Parameters:
        grad_output (numpy.ndarray): Gradient from the next layer (batch_size, input_size).

        Returns:
        numpy.ndarray: Gradient with respect to inputs (batch_size, input_size).
        """
        # compute the sigmoid derivative: ds/dx = s(x) * (1 - s(x))
        sigmoid_grad = self.outputs * (1 - self.outputs)

        # Multiply the sigmoid gradient by the gradient from the next layer
        grad_input = sigmoid_grad * grad_output

        # Return the gradient input.
        return grad_input
