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

class Linear(Layer):
    def __init__(self, input_size, output_size):
        """
        Initializes the Linear layer with random weights and biases.

        Parameters:
        input_size (int):   How many input features.
        output_size (int):  How many output features.
        """
        # Initialize weights to be a Matrix of shape (output_size, input_size) with small random numbers
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        # Initialize bias as a Zero vector of shape (1, output_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        """
        Performs a forward pass of the Linear layer
        
        Parameters:
        inputs (numpy.ndarray):  Input data of shape (batch_size, input_size).

        Returns:
        numpy.ndarray: Output data of shape (batch_size, output_size).
        """
        # Store inputs to be used when doing backpropogation
        self.inputs = inputs

        # print(f"Linear Forward - Input shape: {inputs.shape}, Weights shape: {self.weights.shape}, Bias shape: {self.bias.shape}")

        # Compute and return the forward pass: XW^T + b
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_output):
        """
        Computes gradients for the backward pass.
    
        Parameters:
        grad_output (numpy.ndarray): Gradient from the next layer (batch_size, output_size).

        Returns:
        numpy.ndarray: Gradient with respect to inputs (batch_size, input_size).
        """
        # print(f"Linear Backward - grad_output shape: {grad_output.shape}, Weights shape: {self.weights.shape}")

        # ✅ Always match `grad_weights` shape to `self.weights`
        self.grad_weights = np.dot(self.inputs.T, grad_output)  # Ensures (input_size, output_size)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)  # (1, output_size)

        # ✅ Ensure correct grad_input shape
        grad_input = np.dot(grad_output, self.weights)  # Now (batch_size, input_size)

        # print(f"Updated grad_weights shape: {self.grad_weights.shape}, weights shape: {self.weights.shape}")  # Debugging
        return grad_input

    # def backward(self, grad_output):
        # """
        # Computes gradients for the backward pass.
# 
        # Parameters:
        # grad_output (numpy.ndarray): Gradient from the next layer (batch_size, output_size).
# 
        # Returns:
        # numpy.ndarray: Gradient with respect to inputs (batch_size, input_size).
        # """
        # print(f"Linear Backward - grad_output shape: {grad_output.shape}, Weights shape: {self.weights.shape}")
# 
        # # Gradients of weights and bias
        # # Compute grad_weights: dL/dW = (grad_output^T) * inputs
        # self.grad_weights = np.dot(grad_output.T, self.inputs)
        # # Compute grad_bias: dL/db = sum of grad_output over batch_size
        # self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
# 
        # # Gradient with respect to inputs (to propogate backward)
        # # Compute and return grad_input: dL/dX = grad_output * W
        # grad_input = np.dot(grad_output, self.weights.T)
        # return grad_input