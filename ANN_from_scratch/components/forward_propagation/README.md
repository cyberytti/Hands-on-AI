# Forward Propagation

**Forward propagation** (or the forward pass) is the process of feeding input data forward through the layers of the neural network to compute predicted outputs.

---

## 🔄 Overview & Data Flow

In our 2-layer network architecture, data flows from the 4 input features to the final 3 output probabilities through two main processing stages:

```text
  Input Features (X) 
          │
          ▼
   Linear Step 1: Z1 = W1 · X + b1
          │
          ▼
 Non-Linear Activation: A1 = ReLU(Z1)
          │
          ▼
   Linear Step 2: Z2 = W2 · A1 + b2
          │
          ▼
   Output Activation: A2 = Softmax(Z2)
          │
          ▼
 Predicted Probabilities (A2)
```

Each layer consists of two distinct calculation steps:
1. **Linear Step ($Z$):** Compute weighted sums of inputs plus biases.
2. **Activation Step ($A$):** Apply a non-linear activation function to $Z$.

---

## 🧮 Mathematical Formulas & Matrix Dimensions

Let $m$ denote the number of training samples in batch $X$.

### 1. Hidden Layer Calculations

- **Linear Transformation ($Z_1$):**
  $$Z_1 = W_1 X + b_1$$
  - $W_1 \in \mathbb{R}^{8 \times 4}$
  - $X \in \mathbb{R}^{4 \times m}$
  - $b_1 \in \mathbb{R}^{8 \times 1}$ (broadcasted across all $m$ columns)
  - **Resulting Shape ($Z_1$):** $(8, m)$

- **Activation ($A_1$):**
  $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$
  - **Resulting Shape ($A_1$):** $(8, m)$

---

### 2. Output Layer Calculations

- **Linear Transformation ($Z_2$):**
  $$Z_2 = W_2 A_1 + b_2$$
  - $W_2 \in \mathbb{R}^{3 \times 8}$
  - $A_1 \in \mathbb{R}^{8 \times m}$
  - $b_2 \in \mathbb{R}^{3 \times 1}$ (broadcasted across all $m$ columns)
  - **Resulting Shape ($Z_2$):** $(3, m)$

- **Activation ($A_2$):**
  $$A_2 = \text{softmax}(Z_2)$$
  - **Resulting Shape ($A_2$):** $(3, m)$
  - Each column in $A_2$ contains 3 probability values summing to $1.0$ for the corresponding sample.

---

## 📐 Shape Summary Table

| Matrix | Description | Formula / Source | Shape |
| :--- | :--- | :--- | :--- |
| `X` | Input Feature Matrix | Dataset | `(4, m)` |
| `W1` | Hidden Layer Weights | Random Initialization | `(8, 4)` |
| `b1` | Hidden Layer Biases | Random Initialization | `(8, 1)` |
| `Z1` | Hidden Layer Linear Output | `W1.dot(X) + b1` | `(8, m)` |
| `A1` | Hidden Layer Activated Output | `ReLU(Z1)` | `(8, m)` |
| `W2` | Output Layer Weights | Random Initialization | `(3, 8)` |
| `b2` | Output Layer Biases | Random Initialization | `(3, 1)` |
| `Z2` | Output Layer Linear Output | `W2.dot(A1) + b2` | `(3, m)` |
| `A2` | Network Predictions | `softmax(Z2)` | `(3, m)` |

---

## 🐍 Python Implementation

The forward propagation algorithm is implemented in [`forward_prop.py`](./forward_prop.py):

```python
from components.activation_function.relu_activation import ReLU
from components.activation_function.softmax_activation import softmax

def forward_prop(W1, b1, W2, b2, X):
    """Performs forward propagation."""
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2
```

### Key Implementation Highlights
- `W1.dot(X)` performs matrix multiplication between shape $(8, 4)$ and shape $(4, m)$, resulting in shape $(8, m)$.
- Adding `+ b1` leverages NumPy's broadcasting mechanism to automatically add the $(8, 1)$ bias vector to all $m$ columns of the product matrix.
- Intermediate values ($Z_1, A_1, Z_2, A_2$) are returned because they are required during **backward propagation** to compute gradients!

---

## 🔗 Related Components

- ⬅️ **[Activation Functions](../activation_function/README.md)**
- ➡️ **[Backward Propagation](../backward_propagation/README.md)**
