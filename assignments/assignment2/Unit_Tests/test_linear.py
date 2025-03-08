########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for the Linear Layer implementation.       #
########################################################################

import unittest
import numpy as np
from Layer_Implementations.Linear import Linear

class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        """Initialize a Linear layer."""
        self.layer = Linear(input_size=3, output_size=2)

    def test_forward(self):
        """Test the forward pass of Linear layer."""
        test_input = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        output = self.layer.forward(test_input)
        
        expected_shape = (2, 2)
        self.assertEqual(output.shape, expected_shape, "Linear forward output shape mismatch")

    def test_backward(self):
        """Test the backward pass of Linear layer."""
        test_input = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        self.layer.forward(test_input)

        grad_output = np.array([[0.1, -0.2], [-0.1, 0.2]])
        grad_input = self.layer.backward(grad_output)

        expected_grad_w = np.dot(grad_output.T, test_input)
        expected_grad_b = np.sum(grad_output, axis=0, keepdims=True)

        np.testing.assert_allclose(self.layer.grad_weights, expected_grad_w, atol=1e-6, err_msg="Incorrect weight gradients")
        np.testing.assert_allclose(self.layer.grad_bias, expected_grad_b, atol=1e-6, err_msg="Incorrect bias gradients")

if __name__ == '__main__':
    unittest.main()
