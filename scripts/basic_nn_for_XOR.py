
import sys
import os

# Add ../src to sys.path so we can import modules easily
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "src"))

import numpy as np
# import the neural network implementation
from basic_nn import *

# XOR data
training_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]]))
]
test_data = training_data

# Initialize and train the network
net = Network([2, 3, 1])
net.SGD(training_data, epochs=10000, mini_batch_size=4, learning_rate=1, test_data=test_data)

# Evaluate final predictions
for x, y in test_data:
    output = net.feedforward(x)
    print(f"Input: {x.ravel()}, Predicted: {output[0,0]:.4f}, Actual: {y[0,0]}")
