########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for Binary Cross-Entropy Loss.             #
########################################################################

import unittest
import numpy as np
from Layer_Implementations.BinaryCrossEntropy import BinaryCrossEntropy

class TestBinaryCrossEntropy(unittest.TestCase):
    def setUp(self):
        """Initialize Binary Cross-Entropy loss function."""
        self.loss_fn = BinaryCrossEntropy()
        self.eps = 1e-9

    def test_forward(self):
        """Test the forward pass of BCE."""
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        
        loss = self.loss_fn.forward(y_true, y_pred)
        self.assertGreater(loss, 0, "Loss should be positive")

    def test_backward(self):
        """Test the backward pass of BCE."""
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
        
        grad_output = np.ones_like(y_true)  # Typical for loss functions
        self.loss_fn.forward(y_true, y_pred)  # Forward pass first
        grad = self.loss_fn.backward()

        y_pred_clipped = np.clip(y_pred, self.eps, 1 - self.eps)
        expected_grad = -(y_true / y_pred_clipped) + ((1 - y_true) / (1 - y_pred_clipped))
        
        np.testing.assert_allclose(grad, expected_grad, atol=1e-6, err_msg="BCE gradient is incorrect")

if __name__ == '__main__':
    unittest.main()
