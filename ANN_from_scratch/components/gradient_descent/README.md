# Gradient Descent & Training Loop

**Gradient Descent** is an optimization algorithm used to iteratively update a neural network's parameters (weights and biases) in order to minimize the total loss.

---

## 💡 Difference Between Backpropagation and Gradient Descent

A common point of confusion for beginners is the distinction between backpropagation and gradient descent:

- **Backpropagation:** Calculates *how much* each weight and bias contributed to the prediction error by computing partial derivatives (gradients $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$).
- **Gradient Descent:** Uses those calculated gradients to *actually update* the parameters in the direction that reduces the error.

---

## 🧮 Mathematical Update Rule

The update rules for weights and biases are defined as:

$$W_1 := W_1 - \alpha \cdot dW_1$$

$$b_1 := b_1 - \alpha \cdot db_1$$

$$W_2 := W_2 - \alpha \cdot dW_2$$

$$b_2 := b_2 - \alpha \cdot db_2$$

### Terminology Breakdown
- $W$ and $b$: The current weights and biases.
- $dW$ and $db$: The gradients ($\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$) computed during backpropagation.
- $\alpha$ (**learning rate**): A small positive scalar (e.g., $0.1$ or $0.01$) controlling the step size of each parameter update.
- **Minus Sign ($-$):** Gradients point in the direction of *steepest loss increase*. Subtracting the product $\alpha \cdot dW$ steps parameter values in the exact opposite direction—the direction of *steepest loss decrease*.

---

## ⚙️ Helper Evaluation Functions

To evaluate the performance of our network during training, we implement two helper utilities:

1. **`get_predictions(A)`**: Converts output probability distribution $A_2 \in \mathbb{R}^{3 \times m}$ into predicted class index $\{0, 1, 2\}$ by finding the row index with maximum probability for each sample column:
   $$\hat{y}_i = \arg\max_{k} (A_{2, k, i})$$

2. **`get_accuracy(predictions, Y)`**: Calculates classification accuracy as the fraction of samples correctly predicted:
   $$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}(\hat{y}_i = Y_i)$$

---

## 🔄 Full Training Loop Workflow

The full-batch gradient descent loop ties together parameter initialization, forward propagation, backward propagation, parameter updates, and evaluation:

```text
               Initialize Parameters (W1, b1, W2, b2)
                                 │
                                 ▼
                     ┌──►  Forward Pass
                     │        │  (Z1, A1, Z2, A2)
                     │        ▼
  Repeat Iterations  │    Backward Pass
   (1 to N steps)    │        │  (dW1, db1, dW2, db2)
                     │        ▼
                     │    Update Parameters
                     │        │  (W = W - α * dW)
                     │        ▼
                     └─── Evaluate Accuracy (every 100 iterations)
                                 │
                                 ▼
                    Return Trained Parameters
```

---

## 🐍 Python Implementation

The gradient descent algorithm, parameter updates, and evaluation functions are implemented in [`gradient_descent.py`](./gradient_descent.py):

```python
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
```

---

## 🔗 Related Components

- ⬅️ **[Backward Propagation](../backward_propagation/README.md)**
- 🏠 **[Main Project README](../../README.md)**
- 🚀 **[Main Application Script](../../main.py)**
