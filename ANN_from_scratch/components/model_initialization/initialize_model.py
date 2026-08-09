import numpy as np 

def initialize_parameters():
    """Initializes the parameters for the two-layer neural network."""
    W1 = np.random.rand(8, 4) - 0.5   # Hidden layer: 8 neurons, 4 input features
    b1 = np.random.rand(8, 1) - 0.5   # Hidden layer biases
    W2 = np.random.rand(3, 8) - 0.5   # Output layer: 3 classes, 8 hidden neurons
    b2 = np.random.rand(3, 1) - 0.5   # Output layer biases

    return W1, b1, W2, b2