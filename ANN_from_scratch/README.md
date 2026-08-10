# ANN from Scratch

![Neural network diagram](https://raw.githubusercontent.com/cyberytti/Hands-on-AI/main/ANN_from_scratch/assets/nn.svg)

In this repository, I will walk you through how I coded the neural network shown in the image above from scratch, using only NumPy (no deep learning frameworks such as TensorFlow or PyTorch).

If you are already familiar with Artificial Neural Networks (ANNs), feel free to proceed straight to the code. If not, I recommend reading this article first to build a basic understanding: [ANN – Basic Theory](https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/)

This network is built for the Iris dataset. The Iris dataset contains four input features (sepal length, sepal width, petal length, and petal width) and three possible output classes (three species of iris flowers). For this reason, the input layer has four neurons and the output layer has three neurons.
Since I am building this network from scratch, I have kept the architecture simple: a single hidden layer with eight neurons.

## Weights and Biases

Now, let us look at the weight and bias matrices created for this ANN.

If you are familiar with ANNs, you will know that each neuron has a bias and that each neuron is connected to every neuron in the previous layer (which is why such layers are called _fully connected_ or _dense_ layers). The values traveling along these connections are called signals, and each signal has a weight associated with it. Note, however, that the neurons in the input layer have no weights or biases — weights and biases exist only in the hidden layer(s) and the output layer.

Now, let us calculate how many weights and biases our network needs (that is, the sizes of the matrices).

The input layer has four neurons and the hidden layer has eight neurons. Each of the eight hidden neurons is connected to all four input neurons, which gives us 8 × 4 = 32 weights, plus 8 biases (one for each hidden neuron). In the code, this weight matrix is named `W1` and the bias vector is named `b1`.

The hidden layer has eight neurons and the output layer has three neurons. Each of the three output neurons is connected to all eight hidden neurons, which gives us 3 × 8 = 24 weights, plus 3 biases (one for each output neuron). In the code, this weight matrix is named `W2` and the bias vector is named `b2`.

In total, the network has 32 + 24 = 56 weights and 8 + 3 = 11 biases — that is, 67 parameters that the network must learn.
Below you can see the exact structure of the four matrices used in the code. Here, $w^{(1)}_{ij}$ is the weight connecting input neuron $i$ to hidden neuron $j$, and $w^{(2)}_{ij}$ is the weight connecting hidden neuron $i$ to output neuron $j$.

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

| Matrix | Shape | Description                                                        |
| ------ | ----- | ------------------------------------------------------------------ |
| `W1`   | 4 × 8 | Weights between the input layer and the hidden layer (32 weights)  |
| `b1`   | 8 × 1 | Biases of the hidden layer (8 biases)                              |
| `W2`   | 8 × 3 | Weights between the hidden layer and the output layer (24 weights) |
| `b2`   | 3 × 1 | Biases of the output layer (3 biases)                              |

In the first step, we assign random values to these weights and biases. (We initialize them randomly because if all neurons started with the exact same values, they would all learn the exact same features during training!)

You can find the code implementation for this step here:
👉 **[Model Initialization](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/model_initialization)**

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

You can find the exact code implementation for these matrix operations here:
👉 **[Forward Pass](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/forward_propagation)**

## Activation Functions

An activation function is a mathematical function used to introduce **non-linearity** into a neural network.

Without an activation function, a single neuron's mathematical formula behaves exactly like a simple linear or logistic regression model. If we stacked multiple layers without adding non-linearity, the entire network would just collapse into one large linear equation. Therefore, activation functions are precisely what transform a basic linear model into a powerful deep learning model capable of understanding complex data patterns.

To learn more about activation functions and explore their various types, you can refer to this article: [Activation Functions (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/)

Typically, an activation function is applied at the end of every layer (starting from the hidden layer and ending at the output layer). The choice of which function to use depends on the specific task you are trying to solve.

In this specific neural network, we use two different activation functions:

1. **ReLU (Rectified Linear Unit):** Applied after the hidden layer. It is highly effective and standard for learning complex patterns in intermediate layers.
2. **Softmax:** Applied to the output layer. Because the Iris dataset is a multi-class classification task (predicting one of three flower species), we use Softmax to convert the network's raw output scores into a probability distribution (where all probabilities add up to 1). This pairs perfectly with a cross-entropy loss function.

You can find the exact code implementation of these activation functions here:
👉 **[Activation Functions Implementation](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/activation_function)**

## Loss Function

After completing a forward pass, the network generates a predicted output. We then measure the difference (or error) between this prediction and the actual target values (the true iris species). This measure of error is known as the **loss**, and the specific mathematical formula used to calculate it is called the **loss function**.

To learn more about loss functions and their role in machine learning, you can refer to this resource: [Loss Functions (IBM)](https://www.ibm.com/think/topics/loss-function).

For this specific network, we use the **Cross-Entropy Loss** function. Because we are dealing with a multi-class classification problem (and we used the Softmax activation function in the output layer), Cross-Entropy is the ideal choice to penalize incorrect predictions.

The calculated loss is then used in the next critical phase of training: **backpropagation**. During backpropagation, the network uses this loss value to determine exactly how much each weight contributed to the error, allowing it to adjust the weights and improve for the next pass.

You can find the code implementation for this process here:
👉 **[Backpropagation](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/backward_propagation)**

## Backpropagation

**Backpropagation** (short for "backward propagation of errors") is the fundamental algorithm used to train neural networks. It is defined as the process of calculating the gradient of the loss function with respect to every weight and bias in the network.

After the forward pass generates a prediction and the loss function calculates the error, our goal is to minimize this error to improve the model's accuracy. To achieve this, we propagate the error backward—starting from the output layer and moving toward the input layer.

Using the **chain rule of calculus**, we compute exactly how much each individual weight and bias contributed to the final mistake. These calculated values (called gradients) tell the network the precise direction and magnitude by which it needs to adjust its parameters to become more accurate on the next pass.

To gain a deeper understanding of how backpropagation works both conceptually and mathematically, you can refer to the following resources:

- [Backpropagation Explained (IBM)](https://www.ibm.com/think/topics/backpropagation)
- [Backpropagation in Neural Networks (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/)

You can find the step-by-step mathematical implementation of this process here:
👉 **[Backpropagation Implementation](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/backward_propagation)**
