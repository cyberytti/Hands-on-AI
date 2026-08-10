# Artificial Neural Network (ANN) from Scratch

![Neural Network Diagram](https://raw.githubusercontent.com/cyberytti/Hands-on-AI/main/ANN_from_scratch/assets/nn.svg)

Welcome! In this repository, I will walk you through how I coded the neural network shown in the image above entirely from scratch, using only **NumPy** (no deep learning frameworks like TensorFlow or PyTorch). 

If you are already familiar with Artificial Neural Networks (ANNs), feel free to jump straight to the code. If not, I highly recommend reading this article first to build a foundational understanding: [ANN – Basic Theory (GeeksforGeeks)](https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/).

### The Dataset and Architecture
This network is built to classify the famous **Iris dataset**. The dataset contains four input features (sepal length, sepal width, petal length, and petal width) and three possible output classes (three species of iris flowers). 

Because of this, our architecture requires:
- **Input Layer:** 4 neurons (one for each feature).
- **Hidden Layer:** 8 neurons (to keep the from-scratch implementation simple yet capable of learning patterns).
- **Output Layer:** 3 neurons (one for each flower species).

---

## 📑 Table of Contents
1. [Weights and Biases](#1-weights-and-biases)
2. [Forward Pass](#2-forward-pass)
3. [Activation Functions](#3-activation-functions)
4. [Loss Function](#4-loss-function)
5. [Backpropagation](#5-backpropagation)
6. [Gradient Descent](#6-gradient-descent)
7. [The Training Loop](#7-the-training-loop)

---

## 1. Weights and Biases

Now, let us look at the weight and bias matrices that power this ANN.

If you are familiar with ANNs, you will know that each neuron has a bias and is connected to every neuron in the previous layer (which is why such layers are called *fully connected* or *dense* layers). The values traveling along these connections are called signals, and each signal has a **weight** associated with it. 

*Note: The neurons in the input layer do not have weights or biases — weights and biases exist only in the hidden layer(s) and the output layer.*

### Calculating Matrix Sizes
Let us calculate exactly how many weights and biases our network needs:
- **Input to Hidden:** 8 hidden neurons connected to 4 input neurons = $8 \times 4 = 32$ weights, plus 8 biases. In the code, this weight matrix is `W1` and the bias vector is `b1`.
- **Hidden to Output:** 3 output neurons connected to 8 hidden neurons = $3 \times 8 = 24$ weights, plus 3 biases. In the code, this weight matrix is `W2` and the bias vector is `b2`.

In total, the network has $32 + 24 = 56$ weights and $8 + 3 = 11$ biases — making **67 parameters** that the network must learn.

### Matrix Structures
Below is the exact mathematical structure of the four matrices used in the code. Here, $w^{(1)}_{ij}$ represents the weight connecting input neuron $i$ to hidden neuron $j$.

$$
W_1=
\begin{bmatrix}
w^{(1)}_{11} & w^{(1)}_{12} & w^{(1)}_{13} & w^{(1)}_{14} & w^{(1)}_{15} & w^{(1)}_{16} & w^{(1)}_{17} & w^{(1)}_{18}\\
w^{(1)}_{21} & w^{(1)}_{22} & w^{(1)}_{23} & w^{(1)}_{24} & w^{(1)}_{25} & w^{(1)}_{26} & w^{(1)}_{27} & w^{(1)}_{28}\\
w^{(1)}_{31} & w^{(1)}_{32} & w^{(1)}_{33} & w^{(1)}_{34} & w^{(1)}_{35} & w^{(1)}_{36} & w^{(1)}_{37} & w^{(1)}_{38}\\
w^{(1)}_{41} & w^{(1)}_{42} & w^{(1)}_{43} & w^{(1)}_{44} & w^{(1)}_{45} & w^{(1)}_{46} & w^{(1)}_{47} & w^{(1)}_{48}
\end{bmatrix}
\in \mathbb{R}^{4\times 8}
$$

$$
b_1=
\begin{bmatrix}
b^{(1)}_{1}\\
b^{(1)}_{2}\\
b^{(1)}_{3}\\
b^{(1)}_{4}\\
b^{(1)}_{5}\\
b^{(1)}_{6}\\
b^{(1)}_{7}\\
b^{(1)}_{8}
\end{bmatrix}
\in \mathbb{R}^{8\times 1}
$$

$$
W_2=
\begin{bmatrix}
w^{(2)}_{11} & w^{(2)}_{12} & w^{(2)}_{13}\\
w^{(2)}_{21} & w^{(2)}_{22} & w^{(2)}_{23}\\
w^{(2)}_{31} & w^{(2)}_{32} & w^{(2)}_{33}\\
w^{(2)}_{41} & w^{(2)}_{42} & w^{(2)}_{43}\\
w^{(2)}_{51} & w^{(2)}_{52} & w^{(2)}_{53}\\
w^{(2)}_{61} & w^{(2)}_{62} & w^{(2)}_{63}\\
w^{(2)}_{71} & w^{(2)}_{72} & w^{(2)}_{73}\\
w^{(2)}_{81} & w^{(2)}_{82} & w^{(2)}_{83}
\end{bmatrix}
\in \mathbb{R}^{8\times 3}
$$

$$
b_2=
\begin{bmatrix}
b^{(2)}_{1}\\
b^{(2)}_{2}\\
b^{(2)}_{3}
\end{bmatrix}
\in \mathbb{R}^{3\times 1}
$$

| Matrix | Shape | Description |
|--------|-------|-------------|
| `W1`   | 4 × 8 | Weights between the input layer and the hidden layer (32 weights) |
| `b1`   | 8 × 1 | Biases of the hidden layer (8 biases) |
| `W2`   | 8 × 3 | Weights between the hidden layer and the output layer (24 weights) |
| `b2`   | 3 × 1 | Biases of the output layer (3 biases) |

In the first step, we assign random values to these weights and biases. *(We initialize them randomly because if all neurons started with the exact same values, they would all learn the exact same features during training!)* 

👉 **[Model Initialization Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/model_initialization)**

---

## 2. Forward Pass

Next, we use matrix multiplication to compute the network's output layer by layer. This process of passing data forward through the network is called the **forward pass** (or forward propagation). 

In our code, we break this down into two sub-steps for each layer: 
1. **$Z$ (Linear Step):** We multiply the weights by the inputs and add the bias.
2. **$A$ (Activation Step):** We pass $Z$ through an activation function to introduce non-linearity. $A$ then becomes the input for the next layer.

Here is the exact mathematical flow for our network, where $X$ represents our input data (the four Iris features):

**1. Hidden Layer:**
First, we calculate the linear combination ($Z_1$), and then we apply the **ReLU** activation function to get the hidden layer's output ($A_1$).
$$
Z_1 = W_1 X + b_1
$$
$$
A_1 = \text{ReLU}(Z_1)
$$

**2. Output Layer:**
Next, we use the activated output from the hidden layer ($A_1$) to calculate the linear combination for the output layer ($Z_2$). Finally, because we are classifying the Iris dataset into three distinct species, we apply the **Softmax** function to convert the raw scores into probabilities ($A_2$).
$$
Z_2 = W_2 A_1 + b_2
$$
$$
A_2 = \text{softmax}(Z_2)
$$

👉 **[Forward Pass Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/forward_propagation)**

---

## 3. Activation Functions

An activation function is a mathematical function used to introduce **non-linearity** into a neural network. 

Without an activation function, a single neuron's mathematical formula behaves exactly like a simple linear or logistic regression model. If we stacked multiple layers without adding non-linearity, the entire network would just collapse into one large linear equation. Therefore, activation functions are precisely what transform a basic linear model into a powerful deep learning model capable of understanding complex data patterns.

Typically, an activation function is applied at the end of every layer. In this specific neural network, we use two different activation functions:
1. **ReLU (Rectified Linear Unit):** Applied after the hidden layer. It is highly effective and standard for learning complex patterns in intermediate layers.
2. **Softmax:** Applied to the output layer. Because the Iris dataset is a multi-class classification task, we use Softmax to convert the network's raw output scores into a probability distribution (where all probabilities add up to 1).

👉 **[Activation Functions Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/activation_function)**

---

## 4. Loss Function

After completing a forward pass, the network generates a predicted output. We then measure the difference (or error) between this prediction and the actual target values (the true iris species). This measure of error is known as the **loss**, and the specific mathematical formula used to calculate it is called the **loss function**.

For this specific network, we use the **Cross-Entropy Loss** function. Because we are dealing with a multi-class classification problem (and we used the Softmax activation function in the output layer), Cross-Entropy is the ideal choice to penalize incorrect predictions.

The calculated loss is then used in the next critical phase of training: backpropagation. 

👉 **[Loss Function Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/backward_propagation)**

---

## 5. Backpropagation

**Backpropagation** (short for "backward propagation of errors") is the fundamental algorithm used to train neural networks. It is defined as the process of calculating the gradient of the loss function with respect to every weight and bias in the network.

After the forward pass generates a prediction and the loss function calculates the error, our goal is to minimize this error to improve the model's accuracy. To achieve this, we propagate the error backward—starting from the output layer and moving toward the input layer. 

Using the **chain rule of calculus**, we compute exactly how much each individual weight and bias contributed to the final mistake. These calculated values (called gradients) tell the network the precise direction and magnitude by which it needs to adjust its parameters.

To gain a deeper understanding of how backpropagation works both conceptually and mathematically, you can refer to the following resources:
- [Backpropagation Explained (IBM)](https://www.ibm.com/think/topics/backpropagation)
- [Backpropagation in Neural Networks (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/)

👉 **[Backpropagation Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/backward_propagation)**

---

## 6. Gradient Descent

Gradient descent is an **optimization algorithm** used to iteratively find the optimal weight and bias values that minimize the loss function.

While **backpropagation** calculates *how much* a specific weight contributed to the error (the gradient), **gradient descent** uses that information to actually *update* the weights. You can think of it as walking down a hill blindfolded: the gradient tells you the slope of the hill, and gradient descent dictates taking a step in the downward direction to reach the lowest point (the minimum loss).

The mathematical formulas for updating the weights and biases are:

$$
W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W}
$$

$$
b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial L}{\partial b}
$$

**Where:**
- $W$ and $b$ represent the current weights and biases.
- $\alpha$ (alpha) is the **learning rate**, a small predefined number (e.g., 0.01) that controls the size of the "step" the network takes during the update.
- $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$ are the **gradients** (calculated during backpropagation).

Notice the **minus sign** ($-$) in the formulas. Because the gradient points in the direction of *increasing* loss, we subtract it from the current weights to move in the exact opposite direction, thereby *decreasing* the loss.

👉 **[Gradient Descent Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/gradient_descent)**

---

## 7. The Training Loop

Now that we have implemented the forward pass, loss calculation, backpropagation, and gradient descent, we can combine them into a single training loop.

The training loop is the process that repeatedly performs these steps:
1. **Forward propagation** — generate predictions using the current weights and biases.
2. **Backpropagation** — calculate the gradients of the loss with respect to the weights and biases.
3. **Parameter update** — use gradient descent to update the weights and biases.
4. **Repeat** — continue this process for a fixed number of iterations.

### The Flow
```text
Initialize weights and biases
            ↓
     Forward propagation
            ↓
        Prediction
            ↓
     Backpropagation
            ↓
         Gradients
            ↓
     Update parameters
            ↓
          Repeat
```

### Python Implementation
In our implementation, this process is handled by the `gradient_descent()` function:

```python
def gradient_descent(X, Y, alpha, iterations):
    """Trains the neural network using full-batch gradient descent."""
    # Initialize weights and biases
    W1, b1, W2, b2 = initialize_parameters()
    
    for i in range(1, iterations + 1):
        # 1. Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # 2. Backpropagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        
        # 3. Update parameters using gradient descent
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
    return W1, b1, W2, b2
```

### What happens during one iteration?
Suppose the network starts with some randomly initialized weights and biases. During the first iteration, the input data is passed through the network:
$$ X \rightarrow Z_1 \rightarrow A_1 \rightarrow Z_2 \rightarrow A_2 $$
The output $A_2$ contains the model's predictions. Backpropagation then calculates the gradients, and gradient descent updates the parameters. The next iteration uses these new parameter values to make another prediction. 

With each update, gradient descent attempts to move the parameters toward values that produce a lower loss and, consequently, better predictions. In this project, the network is trained using **full-batch gradient descent**, meaning the entire training dataset is used to calculate the gradients in each iteration.

👉 **[Complete Training Loop Code](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/main.py)**
