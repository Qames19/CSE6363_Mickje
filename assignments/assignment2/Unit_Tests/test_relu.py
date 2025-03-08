########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for the ReLU Layer implementation.         #
########################################################################

import unittest
import numpy as np
from Layer_Implementations.Relu import Relu

class TestReLU(unittest.TestCase):
    def setUp(self):
        """Initialize ReLU layer."""
        self.layer = Relu()

    def test_forward(self):
        """Test the forward pass of ReLU."""
        test_input = np.array([[0.0], [1.0], [-1.0], [10.0], [-10.0]])
        expected_output = np.maximum(0, test_input)

        output = self.layer.forward(test_input)
        np.testing.assert_array_equal(output, expected_output, "ReLU forward pass is incorrect")

    def test_backward(self):
        """Test the backward pass of ReLU."""
        test_input = np.array([[0.0], [1.0], [-1.0], [10.0], [-10.0]])
        self.layer.forward(test_input)
        
        grad_output = np.array([[1.0], [0.5], [-0.5], [0.1], [-0.1]])
        expected_grad = (test_input > 0).astype(float) * grad_output
        
        grad_input = self.layer.backward(grad_output)
        np.testing.assert_array_equal(grad_input, expected_grad, "ReLU backward pass is incorrect")

if __name__ == '__main__':
    unittest.main()
