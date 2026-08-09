import numpy as np

def ReLU(Z):
    """ReLU activation function."""
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    """Derivative of ReLU."""
    return Z > 0
