# Model Initialization

In this component, we initialize the trainable parameters—the weights ($W$) and biases ($b$)—of our Artificial Neural Network.

## 🎯 Purpose

Before training begins, all weights and biases in the network must be assigned starting values. 

If all weights were initialized to zero or the exact same value, every neuron in a given layer would compute the exact same output during forward propagation and receive the exact same gradients during backpropagation. This issue is known as **symmetry**. 

To break symmetry and allow each neuron to learn different features from the data, we initialize the weights randomly.

---

## 📐 Network Architecture & Parameter Shapes

Our neural network is built for the **Iris dataset**:
- **Input Layer:** 4 neurons (corresponding to 4 input features: sepal length, sepal width, petal length, petal width).
- **Hidden Layer:** 8 neurons (Dense / Fully Connected with ReLU activation).
- **Output Layer:** 3 neurons (corresponding to 3 flower species: *Setosa*, *Versicolor*, *Virginica*).

### Parameter Shapes & Quantities

| Parameter | Shape | Size | Description |
| :--- | :--- | :--- | :--- |
| `W1` | `(8, 4)` | $8 \times 4 = 32$ | Weights connecting 4 input features to 8 hidden neurons |
| `b1` | `(8, 1)` | $8 \times 1 = 8$ | Biases for the 8 hidden layer neurons |
| `W2` | `(3, 8)` | $3 \times 8 = 24$ | Weights connecting 8 hidden neurons to 3 output neurons |
| `b2` | `(3, 1)` | $3 \times 1 = 3$ | Biases for the 3 output layer neurons |

**Total Parameters to Learn:** $32 + 8 + 24 + 3 = 67$ parameters.

---

## 🧮 Mathematical Notation

The initialized weight and bias matrices are represented as:

$$
W_1 = \begin{bmatrix}
w^{(1)}_{11} & w^{(1)}_{12} & w^{(1)}_{13} & w^{(1)}_{14} \\
w^{(1)}_{21} & w^{(1)}_{22} & w^{(1)}_{23} & w^{(1)}_{24} \\
\vdots & \vdots & \vdots & \vdots \\
w^{(1)}_{81} & w^{(1)}_{82} & w^{(1)}_{83} & w^{(1)}_{84}
\end{bmatrix} \in \mathbb{R}^{8 \times 4}, \quad
b_1 = \begin{bmatrix}
b^{(1)}_1 \\ b^{(1)}_2 \\ \vdots \\ b^{(1)}_8
\end{bmatrix} \in \mathbb{R}^{8 \times 1}
$$

$$
W_2 = \begin{bmatrix}
w^{(2)}_{11} & w^{(2)}_{12} & \cdots & w^{(2)}_{18} \\
w^{(2)}_{21} & w^{(2)}_{22} & \cdots & w^{(2)}_{28} \\
w^{(2)}_{31} & w^{(2)}_{32} & \cdots & w^{(2)}_{38}
\end{bmatrix} \in \mathbb{R}^{3 \times 8}, \quad
b_2 = \begin{bmatrix}
b^{(2)}_1 \\ b^{(2)}_2 \\ b^{(2)}_3
\end{bmatrix} \in \mathbb{R}^{3 \times 1}
$$

---

## 🐍 Python Implementation

The parameter initialization is implemented in [`initialize_model.py`](./initialize_model.py):

```python
import numpy as np 

def initialize_parameters():
    """Initializes the parameters for the two-layer neural network."""
    W1 = np.random.rand(8, 4) - 0.5   # Hidden layer: 8 neurons, 4 input features
    b1 = np.random.rand(8, 1) - 0.5   # Hidden layer biases
    W2 = np.random.rand(3, 8) - 0.5   # Output layer: 3 classes, 8 hidden neurons
    b2 = np.random.rand(3, 1) - 0.5   # Output layer biases

    return W1, b1, W2, b2
```

### Explanation of Implementation Details

1. `np.random.rand(shape)` generates numbers uniformly distributed in the interval $[0, 1)$.
2. Subtracting `0.5` shifts this range to $[-0.5, 0.5)$. Centering initial weights around zero helps prevent initial layer outputs from becoming overly large or biased toward positive numbers.

---

## 🔗 Related Components

- ⬅️ **[Main Project README](../../README.md)**
- ➡️ **Next Step:** **[Forward Propagation](../forward_propagation/README.md)**
