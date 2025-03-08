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

class Relu(Layer):
    def __init__(self):
        """
        Initializes the ReLU activation layer        
        """
        pass

    def forward(self, inputs):
        """
        Performs a forward pass of the ReLU layer.

        Parameters:
        inputs (numpy.ndarray): input data.

        Returns:
        numpy.ndarray: output values of max(0, input).
        """
        # Compute and return ReLU forward pass ReLU(input) = max(0, input)
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        """
        Performs a backward pass of the ReLU layer.

        Parameters:
        grad_output (numpy.ndarray): Gradients from the next layer (batch_size, input_size).

        Returns:
        numpy.ndarray: Gradient with respect to inputs (batch_size, input_size).
        
        """
        # Compute gradient of ReLU: dReLU/dx = {1, x > 0 ; 0, x <= 0}
        relu_grad = np.where(self.inputs > 0, 1, 0)

        # Multiply the relu_grad by the grad_input
        grad_input = grad_output * relu_grad

        # Return grad_input
        return grad_input