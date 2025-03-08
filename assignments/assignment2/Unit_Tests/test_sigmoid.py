########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for the Sigmoid Layer implementation.      #
########################################################################

import unittest
import numpy as np
from Layer_Implementations.Sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        """Initialize Sigmoid layer."""
        self.layer = Sigmoid()

    def test_forward(self):
        """Test the forward pass of Sigmoid."""
        test_input = np.array([[0.0], [1.0], [-1.0], [10.0], [-10.0]])
        expected_output = 1 / (1 + np.exp(-test_input))

        output = self.layer.forward(test_input)
        np.testing.assert_allclose(output, expected_output, atol=1e-6, err_msg="Sigmoid forward pass is incorrect")

    def test_backward(self):
        """Test the backward pass of Sigmoid."""
        test_input = np.array([[0.0], [1.0], [-1.0], [10.0], [-10.0]])
        self.layer.forward(test_input)  # Store output for backpropagation

        grad_output = np.array([[1.0], [0.5], [-0.5], [0.1], [-0.1]])
        expected_grad = (self.layer.outputs * (1 - self.layer.outputs)) * grad_output

        grad_input = self.layer.backward(grad_output)
        np.testing.assert_allclose(grad_input, expected_grad, atol=1e-6, err_msg="Sigmoid backward pass is incorrect")

if __name__ == '__main__':
    unittest.main()
