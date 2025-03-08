########################################################################
# Author:       ChatGPT (o3-mini-high)                                 #
# Assisted By:  James Mick                                             #
# Date:         02/16/2025                                             #
# Description:  Test script for the Relu Layer implementation.         #
########################################################################

import unittest
import numpy as np
import os
from Layer_Implementations.Sequential import Sequential
from Layer_Implementations.Linear import Linear
from Layer_Implementations.Relu import Relu

class TestModelSaving(unittest.TestCase):
    def setUp(self):
        """Initialize a simple model."""
        self.model = Sequential()
        self.model.add(Linear(input_size=3, output_size=4))
        self.model.add(Relu())
        self.model.add(Linear(input_size=4, output_size=2))

        self.test_file = "test_model_weights.npz"

    def test_save_and_load_weights(self):
        """Test saving and loading model weights."""
        # Save initial weights
        self.model.save_weights(self.test_file)

        # Create a new model and load weights
        new_model = Sequential()
        new_model.add(Linear(input_size=3, output_size=4))
        new_model.add(Relu())
        new_model.add(Linear(input_size=4, output_size=2))

        new_model.load_weights(self.test_file)

        # Compare weights
        for layer1, layer2 in zip(self.model.layers, new_model.layers):
            if hasattr(layer1, 'weights') and hasattr(layer1, 'bias'):
                np.testing.assert_array_equal(layer1.weights, layer2.weights, "Weights mismatch")
                np.testing.assert_array_equal(layer1.bias, layer2.bias, "Bias mismatch")

        print("✅ Model weights saved and loaded successfully!")

    def tearDown(self):
        """Remove the test file after the test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()
