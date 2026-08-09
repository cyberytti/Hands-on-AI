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

Next, we use the matrix multiplication formulas discussed earlier to compute the network's output layer by layer. This process of passing data through the network is called the **forward pass** (or forward propagation).

You can find the code implementation for this step here:
👉 **[Forward Pass](https://github.com/cyberytti/Hands-on-AI/tree/main/ANN_from_scratch/components/forward_propagation)**
