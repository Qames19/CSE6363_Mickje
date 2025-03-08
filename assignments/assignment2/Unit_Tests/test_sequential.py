########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for the Sequential Model.                  #
########################################################################

import unittest
import numpy as np
from Layer_Implementations.Sequential import Sequential
from Layer_Implementations.Linear import Linear
from Layer_Implementations.Relu import Relu
from Layer_Implementations.Sigmoid import Sigmoid

class TestSequentialModel(unittest.TestCase):
    def setUp(self):
        """Initialize a Sequential model."""
        self.model = Sequential()
        self.model.add(Linear(input_size=3, output_size=4))
        self.model.add(Relu())
        self.model.add(Linear(input_size=4, output_size=2))
        self.model.add(Sigmoid())

    def test_forward(self):
        """Test forward pass through the entire model."""
        X_test = np.array([[0.5, -0.2, 1.0], [-1.5, 2.0, -0.5]])
        output = self.model.forward(X_test)

        expected_shape = (2, 2)
        self.assertEqual(output.shape, expected_shape, "Sequential forward pass shape mismatch")

    def test_backward(self):
        """Test backward pass through the model."""
        X_test = np.array([[0.5, -0.2, 1.0], [-1.5, 2.0, -0.5]])
        self.model.forward(X_test)

        grad_output = np.ones_like(self.model.layers[-1].outputs)  # Typically all ones
        grad_input = self.model.backward(grad_output)

        self.assertEqual(grad_input.shape, X_test.shape, "Backpropagation output shape mismatch")

if __name__ == '__main__':
    unittest.main()
