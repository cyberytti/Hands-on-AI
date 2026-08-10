# Activation Functions

Activation functions introduce **non-linearity** into artificial neural networks. Without non-linear activation functions, stacking multiple dense layers would mathematically collapse into a single linear transformation ($W_{net} X + b_{net}$), making it impossible for the network to learn complex patterns.

This directory contains two key activation functions used in our network:
1. **ReLU (Rectified Linear Unit)** — Used in the hidden layer.
2. **Softmax** — Used in the output layer for multi-class classification.

---

## ⚡ 1. ReLU (Rectified Linear Unit)

### Concept & Mathematical Definition

The **ReLU** function outputs the input directly if it is positive; otherwise, it outputs zero. It is simple, computationally efficient, and helps prevent the vanishing gradient problem during backpropagation.

$$
\text{ReLU}(Z) = \max(0, Z)
$$

### Derivative of ReLU

During backpropagation, we need the derivative of ReLU with respect to $Z$ to compute gradients for the hidden layer:

$$
\text{ReLU}'(Z) = \begin{cases} 
1 & \text{if } Z > 0 \\ 
0 & \text{if } Z \le 0 
\end{cases}
$$

### 🐍 Python Implementation

The ReLU function and its derivative are implemented in [`relu_activation.py`](./relu_activation.py):

```python
import numpy as np

def ReLU(Z):
    """ReLU activation function."""
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    """Derivative of ReLU."""
    return Z > 0
```

- `np.maximum(Z, 0)` element-wise replaces negative numbers in matrix $Z$ with `0`.
- `Z > 0` produces a boolean array (`True` where $Z > 0$, `False` elsewhere), which NumPy automatically converts to $1$ and $0$ during arithmetic multiplication in backpropagation.

---

## 🎯 2. Softmax Activation

### Concept & Mathematical Definition

The **Softmax** function is applied at the output layer of multi-class classification networks. It takes raw, unnormalized linear scores ($Z_2$) and transforms them into a probability distribution where each value lies between $0$ and $1$, and the sum of all class probabilities equals $1$.

For a vector $Z = [z_1, z_2, \dots, z_K]^T$ with $K$ classes:

$$
\text{softmax}(Z)_i = \frac{e^{Z_i}}{\sum_{j=1}^{K} e^{Z_j}} \quad \text{for } i = 1, \dots, K
$$

### 🛡️ Numerical Stability Trick

Evaluating $e^{Z_i}$ directly can lead to floating-point overflow error if $Z_i$ contains large positive numbers (e.g., $e^{1000} \to \infty$).

To make Softmax numerically stable, we subtract the maximum value in $Z$ from all entries prior to exponentiation:

$$
Z_{\text{stable}} = Z - \max(Z)
$$

$$
\text{softmax}(Z)_i = \frac{e^{Z_{\text{stable}, i}}}{\sum_{j=1}^{K} e^{Z_{\text{stable}, j}}}
$$

Because subtracting a constant from all inputs scales numerator and denominator by $e^{-\max(Z)}$, the output values remain identical while guaranteeing that the largest exponent evaluated is $e^0 = 1$.

### 🐍 Python Implementation

The Softmax function is implemented in [`softmax_activation.py`](./softmax_activation.py):

```python
import numpy as np

def softmax(Z):
    """Numerically stable softmax for each column."""
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
```

- `axis=0` operates along columns (since each column represents an individual sample when data is stored as shape $(N_{features}, m)$).
- `keepdims=True` preserves matrix dimensions to ensure correct broadcasting during subtraction and division.

---

## 📊 Summary Comparison

| Activation Function | Layer Applied | Output Range | Main Purpose |
| :--- | :--- | :--- | :--- |
| **ReLU** | Hidden Layer | $[0, \infty)$ | Introduces non-linearity; enables deep representation learning |
| **Softmax** | Output Layer | $(0, 1)$ with $\sum P = 1$ | Converts raw outputs into class probability distributions |

---

## 🔗 Related Components

- ⬅️ **[Model Initialization](../model_initialization/README.md)**
- ➡️ **[Forward Propagation](../forward_propagation/README.md)**
