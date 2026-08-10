# Backward Propagation

**Backpropagation** (short for _backward propagation of errors_) is the core algorithm used to train neural networks. It calculates the gradient of the loss function with respect to every weight and bias in the network using the **Chain Rule of Calculus**.

---

## 🎯 Purpose

During forward propagation, the network generates predictions A<sub>2</sub>. Comparing A<sub>2</sub> against the ground truth labels Y allows us to calculate the loss (error).

Backpropagation moves backward from the output layer to the hidden layer, calculating how much each weight (W<sub>1</sub>, W<sub>2</sub>) and bias (b<sub>1</sub>, b<sub>2</sub>) contributed to that loss. These calculated rates of change—called gradients (∂L/∂W, ∂L/∂b)—tell us the direction and magnitude to update parameters during optimization.

---

## 🏷️ Target Label One-Hot Encoding

The target species vector Y contains integer class labels {0, 1, 2} for each sample (shape (m,)). To compare predictions A<sub>2</sub> ∈ R<sup>3 × m</sup> directly against ground truth, we convert Y into a binary one-hot matrix Y<sub>one_hot</sub> ∈ R<sup>3 × m</sup>.

For example, if sample i has label 1 (Versicolor):  
Y<sub>one_hot</sub>[:, i] = [0, 1, 0]<sup>T</sup>

---

## 🧮 Step-by-Step Mathematical Derivations

### 1. Output Layer Gradients

#### Error at Output Layer (dZ<sub>2</sub>)

When combining categorical cross-entropy loss with the softmax activation function, the partial derivative of the loss with respect to Z<sub>2</sub> simplifies elegantly to:

dZ<sub>2</sub> = A<sub>2</sub> - Y<sub>one_hot</sub> ∈ R<sup>3 × m</sup>

#### Gradients for Output Weights (dW<sub>2</sub>) and Biases (db<sub>2</sub>)

Applying the chain rule:

dW<sub>2</sub> = (1/m) · dZ<sub>2</sub> A<sub>1</sub><sup>T</sup> ∈ R<sup>3 × 8</sup>

db<sub>2</sub> = (1/m) · Σ dZ<sub>2</sub> ∈ R<sup>3 × 1</sup>

- Dividing by m computes the average gradient across all m samples in the batch.

---

### 2. Hidden Layer Gradients

#### Error at Hidden Layer (dZ<sub>1</sub>)

To backpropagate error from the output layer to the hidden layer, we multiply dZ<sub>2</sub> by the transpose of W<sub>2</sub> and apply element-wise multiplication (⊙) with the derivative of the ReLU function:

dZ<sub>1</sub> = (W<sub>2</sub><sup>T</sup> dZ<sub>2</sub>) ⊙ ReLU'(Z<sub>1</sub>) ∈ R<sup>8 × m</sup>

where ReLU'(Z<sub>1</sub>) = 1 if Z<sub>1</sub> > 0, and 0 otherwise.

#### Gradients for Hidden Weights (dW<sub>1</sub>) and Biases (db<sub>1</sub>)

Applying the chain rule again:

dW<sub>1</sub> = (1/m) · dZ<sub>1</sub> X<sup>T</sup> ∈ R<sup>8 × 4</sup>

db<sub>1</sub> = (1/m) · Σ dZ<sub>1</sub> ∈ R<sup>8 × 1</sup>

---

## 📐 Matrix Dimensions Summary Table

| Gradient Variable | Formula / Derivation             | Shape    | Description                                     |
| :---------------- | :------------------------------- | :------- | :---------------------------------------------- |
| `one_hot_Y`       | `one_hot(Y)`                     | `(3, m)` | One-hot binary indicator matrix of true targets |
| `dZ2`             | `A2 - one_hot_Y`                 | `(3, m)` | Output layer prediction error                   |
| `dW2`             | `(1 / m) * dZ2.dot(A1.T)`        | `(3, 8)` | Gradient of loss w.r.t output weights $W_2$     |
| `db2`             | `(1 / m) * np.sum(dZ2, axis=1)`  | `(3, 1)` | Gradient of loss w.r.t output biases $b_2$      |
| `dZ1`             | `W2.T.dot(dZ2) * ReLU_deriv(Z1)` | `(8, m)` | Hidden layer error propagated from output       |
| `dW1`             | `(1 / m) * dZ1.dot(X.T)`         | `(8, 4)` | Gradient of loss w.r.t hidden weights $W_1$     |
| `db1`             | `(1 / m) * np.sum(dZ1, axis=1)`  | `(8, 1)` | Gradient of loss w.r.t hidden biases $b_1$      |

---

## 🐍 Python Implementation

The backward propagation algorithms are implemented in [`backward_prop.py`](./backward_prop.py):

```python
import numpy as np
from components.activation_function.relu_activation import ReLU_deriv

def one_hot(Y):
    """Converts label vector into one-hot encoded matrix."""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """Performs backward propagation and computes gradients."""
    m = Y.size
    one_hot_Y = one_hot(Y)

    # Output layer gradients
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradients
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2
```

---

## 🔗 Related Components

- ⬅️ **[Forward Propagation](../forward_propagation/README.md)**
- ➡️ **[Gradient Descent](../gradient_descent/README.md)**
