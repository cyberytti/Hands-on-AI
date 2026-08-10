# Neural Network Components

This directory contains the modular building blocks of our Artificial Neural Network. By separating the network into distinct, single-responsibility components, the code remains clean, readable, and easy to debug.

## 📂 Directory Structure

- **[model_initialization](./model_initialization/)**: Handles the random initialization of weights and biases to break symmetry.
- **[activation_function](./activation_function/)**: Contains the ReLU and Softmax functions used to introduce non-linearity.
- **[forward_propagation](./forward_propagation/)**: Computes the network's predictions layer by layer.
- **[backward_propagation](./backward_propagation/)**: Calculates the gradients of the loss with respect to the parameters using the chain rule.
- **[gradient_descent](./gradient_descent/)**: Updates the weights and biases using the calculated gradients.

Navigate into any of the folders above to see the mathematical theory and the specific Python implementation for that step!
