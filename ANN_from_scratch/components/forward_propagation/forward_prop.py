from components.activation_function.relu_activation import ReLU
from components.activation_function.softmax_activation import softmax

def forward_prop(W1, b1, W2, b2, X):
    """Performs forward propagation."""
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2