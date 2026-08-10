# Backward Propagation

**Backpropagation** (short for *backward propagation of errors*) is the core algorithm used to train neural networks. It calculates the gradient of the loss function with respect to every weight and bias in the network using the **Chain Rule of Calculus**.

---

## 🎯 Purpose

During forward propagation, the network generates predictions $A_2$. Comparing $A_2$ against the ground truth labels $Y$ allows us to calculate the loss (error). 

Backpropagation moves backward from the output layer to the hidden layer, calculating how much each weight ($W_1, W_2$) and bias ($b_1, b_2$) contributed to that loss. These calculated rates of change—called **gradients** ($\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$)—tell us the direction and magnitude to update parameters during optimization.

---

## 🏷️ Target Label One-Hot Encoding

The target species vector $Y$ contains integer class labels $\{0, 1, 2\}$ for each sample (shape `(m,)`). To compare predictions $A_2 \in \mathbb{R}^{3 \times m}$ directly against ground truth, we convert $Y$ into a binary **one-hot matrix** $Y_{\text{one\_hot}} \in \mathbb{R}^{3 \times m}$.

For example, if sample $i$ has label $1$ (*Versicolor*):
$$Y_{\text{one\_hot}}[:, i] = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

---

## 🧮 Step-by-Step Mathematical Derivations

### 1. Output Layer Gradients

#### Error at Output Layer ($dZ_2$)
When combining Categorical Cross-Entropy loss with the Softmax activation function, the partial derivative of the loss with respect to $Z_2$ simplifies elegantly to:

$$dZ_2 = \frac{\partial L}{\partial Z_2} = A_2 - Y_{\text{one\_hot}} \quad \in \mathbb{R}^{3 \times m}$$

#### Gradients for Output Weights ($dW_2$) and Biases ($db_2$)
Applying the chain rule:

$$dW_2 = \frac{\partial L}{\partial W_2} = \frac{1}{m} dZ_2 A_1^T \quad \in \mathbb{R}^{3 \times 8}$$

$$db_2 = \frac{\partial L}{\partial b_2} = \frac{1}{m} \sum_{i=1}^{m} dZ_2 \quad \in \mathbb{R}^{3 \times 1}$$

- Dividing by $m$ computes the average gradient across all $m$ samples in the batch.

---

### 2. Hidden Layer Gradients

#### Error at Hidden Layer ($dZ_1$)
To backpropagate error from the output layer to the hidden layer, we multiply $dZ_2$ by the transpose of $W_2$ and apply element-wise multiplication ($\odot$) with the derivative of the ReLU function:

$$dZ_1 = \frac{\partial L}{\partial Z_1} = (W_2^T dZ_2) \odot \text{ReLU}'(Z_1) \quad \in \mathbb{R}^{8 \times m}$$

where $\text{ReLU}'(Z_1) = 1$ if $Z_1 > 0$, and $0$ otherwise.

#### Gradients for Hidden Weights ($dW_1$) and Biases ($db_1$)
Applying the chain rule again:

$$dW_1 = \frac{\partial L}{\partial W_1} = \frac{1}{m} dZ_1 X^T \quad \in \mathbb{R}^{8 \times 4}$$

$$db_1 = \frac{\partial L}{\partial b_1} = \frac{1}{m} \sum_{i=1}^{m} dZ_1 \quad \in \mathbb{R}^{8 \times 1}$$

---

## 📐 Matrix Dimensions Summary Table

| Gradient Variable | Formula / Derivation | Shape | Description |
| :--- | :--- | :--- | :--- |
| `one_hot_Y` | `one_hot(Y)` | `(3, m)` | One-hot binary indicator matrix of true targets |
| `dZ2` | `A2 - one_hot_Y` | `(3, m)` | Output layer prediction error |
| `dW2` | `(1 / m) * dZ2.dot(A1.T)` | `(3, 8)` | Gradient of loss w.r.t output weights $W_2$ |
| `db2` | `(1 / m) * np.sum(dZ2, axis=1)` | `(3, 1)` | Gradient of loss w.r.t output biases $b_2$ |
| `dZ1` | `W2.T.dot(dZ2) * ReLU_deriv(Z1)` | `(8, m)` | Hidden layer error propagated from output |
| `dW1` | `(1 / m) * dZ1.dot(X.T)` | `(8, 4)` | Gradient of loss w.r.t hidden weights $W_1$ |
| `db1` | `(1 / m) * np.sum(dZ1, axis=1)` | `(8, 1)` | Gradient of loss w.r.t hidden biases $b_1$ |

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
