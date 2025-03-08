########################################################################
# Student:      James Mick                                             #
# ID:           2234431                                                #
# Course:       CSE-6363                                               #
# Assignment:   2                                                      #
# Professor:    Dr. Alex Dilhoff                                       #
# Assisted by:  ChatGPT 4o                                             #
########################################################################

import os
import numpy as np
from Layer_Implementations.Sequential import Sequential
from Layer_Implementations.Linear import Linear
from Layer_Implementations.Sigmoid import Sigmoid
from Layer_Implementations.BinaryCrossEntropy import BinaryCrossEntropy

# Define the XOR Dataset
inputs  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Define the Model
model = Sequential()
model.add(Linear(2,2))
model.add(Sigmoid())
model.add(Linear(2,1))
model.add(Sigmoid())

# Define the Loss function
bcel = BinaryCrossEntropy()

# Define the HyperParameters
lr = 1        #learning_rate
num_epochs = 5000   #Number of iterations to train

# Train the model
for epoch in range(num_epochs):
    # Do a forward pass to obtain predictions
    predictions = model.forward(inputs)

    # Compute the loss
    loss = bcel.forward(outputs, predictions)
    grad_loss = bcel.backward()

    # Do a backward pass to compute gradients
    model.backward(grad_loss)

    # Update weights for Gradient Descent
    for layer in model.layers:
        if hasattr(layer, "weights") and hasattr(layer, "bias"):    # Only update on trainable layers
            layer.weights -= lr * layer.grad_weights
            layer.bias -= lr * layer.grad_bias
    
    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
# define filename for read/write
file_path = "XOR_solved.w"

# Save the trained model weights
model.save_weights(file_path)
print("XOR Model trained and saved")

# Load the trained model
model.load_weights(file_path)

# Test on XOR inputs
test_outputs = model.forward(inputs)

# Print the predictions
print("XOR Predictions:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Predicted Output: {test_outputs[i][0]:.1f}")