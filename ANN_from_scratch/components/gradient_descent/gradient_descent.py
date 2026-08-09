import numpy as np
from components.forward_propagation.forward_prop import forward_prop
from components.backward_propagation.backward_prop import backward_prop
from components.model_initialization.initialize_model import initialize_parameters

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """Updates parameters using gradient descent."""
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predictions(A):
    """Returns predicted class labels."""
    return np.argmax(A, axis=0)


def get_accuracy(predictions, Y):
    """Returns classification accuracy."""
    return np.mean(predictions == Y)


def gradient_descent(X, Y, alpha, iterations):
    """Trains the neural network using full-batch gradient descent."""
    W1, b1, W2, b2 = initialize_parameters()

    for i in range(1, iterations + 1):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Parameter updates
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2,
            dW1, db1, dW2, db2,
            alpha
        )

        # Print training accuracy every 100 iterations
        if i % 100 == 0:
            _, _, _, A2_eval = forward_prop(W1, b1, W2, b2, X)
            predictions = get_predictions(A2_eval)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}, Training Accuracy: {accuracy:.4f}")

    return W1, b1, W2, b2