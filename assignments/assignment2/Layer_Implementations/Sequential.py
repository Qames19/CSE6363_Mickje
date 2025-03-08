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

class Sequential(Layer):
    def __init__(self):
        """
        Initialize a neural network composed of multiple activation layers
        """
        self.layers = []

    def forward(self, inputs):
        """
        Performs forward pass through a sequential list of layers in the model.

        Parameters:
        inputs (np.ndarray): Input data.

        Returns:
        np.ndarray: The output after passing through each layer in the model.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        return inputs

    def backward(self, grad_output):
        """
        Performs backpropation through a sequential list of layers in the model.

        Parameters:
        grad_output (np.ndarray): Gradient from the next layer in the model (batch_size, input_size).

        Returns:
        np.ndarray: Gradient with respect to inputs (batch_size, input_size). 
        """

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        
        return grad_output

    def add(self, layer):
        """
        Adds a layer to the model.

        Parameters:
        layer (Layer): A layer (Linear, Sigmoid, ReLU, BinaryCrossEntropy, etc.)
        """
        # append a layer to the network
        self.layers.append(layer)
    
    def save_weights(self, file_path):
        """
        Saves weights and biases as a binary file.

        Parameters:
        file_path (str): Path to save the weights.
        """
        with open(file_path, "wb") as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "weights") and hasattr(layer, "bias"):
                    np.save(f, layer.weights)
                    np.save(f, layer.bias)
            print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path):
        """
        Loads weights and biases from a binary file.

        Parameters:
        file_path (str): Path to load the weights from.
        """
        with open(file_path, "rb") as f:
            for layer in self.layers:
                if hasattr(layer, "weights") and hasattr(layer, "bias"):
                    layer.weights = np.load(f)
                    layer.bias = np.load(f)
            data = np.load(file_path)
        print(f"Model weights loaded from {file_path}")