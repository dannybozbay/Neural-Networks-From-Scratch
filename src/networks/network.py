"""
network.py
~~~~~~~~~~

A module implementing a simple feedforward neural network, trained using
the stochastic gradient descent (SGD) learning algorithm. This implementation
calculates gradients via backpropagation, enabling the network to learn from
data efficiently.

Key Features:
- Feedforward propagation to compute activations across layers.
- Stochastic gradient descent for training the network with mini-batches.
- Backpropagation to calculate gradients for the cost function, updating weights
  and biases accordingly.

This code is designed for simplicity, readability, and ease of modification. It
is intended for educational purposes and basic experimentation. As such, it is
not optimized for performance and lacks advanced features like regularization,
early stopping, and specialized optimization algorithms.
"""

# Libraries
import random
import time

import numpy as np


class Network:
    """
    A class representing a neural network for feedforward propagation and stochastic gradient descent training.

    Attributes:
    - sizes (list of int): List containing the number of neurons in each layer of the network.
    - num_layers (int): Number of layers in the network, including input and output layers.
    - biases (list of ndarray): List of bias vectors for each layer (excluding the input layer).
    - weights (list of ndarray): List of weight matrices for each layer (connections between consecutive layers).

    Methods:
    - __init__(sizes): Initialize the network with given layer sizes.
    - feedforward(a): Perform feedforward propagation to compute activations for a given input.
    - SGD(training_data, epochs, mini_batch_size, eta, validation_data=None, monitor_training_accuracy=False, monitor_validation_accuracy=False): Train the network using mini-batch stochastic gradient descent.
    - update_mini_batch(mini_batch, eta): Update weights and biases using backpropagation for a given mini-batch.
    - backprop(x, y): Compute gradients for the cost function using backpropagation for a single training example.
    - accuracy(data, convert=False): Calculate the accuracy of the network on the given data.
    - cost_derivative(output_activations, y): Compute the derivative of the cost function with respect to output activations.
    """

    def __init__(self, sizes):
        """
        Initialize the neural network with given layer sizes.

        Parameters:
        - sizes (list of int): List containing the number of neurons in each layer, including input and output layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for all layers except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights for all layers (connections between layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Perform feedforward propagation, computing activations using weights and biases.

        Parameters:
        - a (ndarray): Input activation vector for the network.

        Returns:
        - ndarray: Output activation vector after propagation through the network.
        """
        # Iterate through each layer, applying weights, biases, and the sigmoid function
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        validation_data=None,
        monitor_training_accuracy=False,
        monitor_validation_accuracy=False,
    ):
        """
        Train the neural network using stochastic gradient descent.

        Parameters:
        - training_data (list of tuples): List of tuples (x, y) representing training inputs and corresponding target outputs.
        - epochs (int): Number of training epochs.
        - mini_batch_size (int): Size of each mini-batch for stochastic gradient descent.
        - eta (float): Learning rate.
        - validation_data (list of tuples, optional): If provided, the network will be evaluated against validation data after each epoch.
        - monitor_training_accuracy (bool): If True, the accuracy on training data will be monitored after each epoch.
        - monitor_validation_accuracy (bool): If True, the accuracy on validation data will be monitored after each epoch.

        Returns:
        - tuple: Lists containing training accuracy and validation accuracy per epoch (if monitored).
        """
        n_train = len(training_data)
        training_acc, validation_acc = [], []
        t0 = time.time()  # Track the start time for epoch duration measurement
        for j in range(epochs):
            random.shuffle(training_data)  # Shuffle training data for each epoch
            # Split training data into mini-batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)
            ]
            # Update network for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")
            # Evaluate and print the network's performance
            INDENT = " " * 4
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_acc.append(accuracy)
                print(f"{INDENT}Accuracy on training data: {accuracy:.2f}")
            if monitor_validation_accuracy:
                accuracy = self.accuracy(validation_data, convert=False)
                validation_acc.append(accuracy)
                print(f"{INDENT}Accuracy on validation data: {accuracy:.2f}")

        return training_acc, validation_acc

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        Parameters:
        - mini_batch (list of tuples): List of tuples (x, y) representing mini-batch inputs and corresponding target outputs.
        - eta (float): Learning rate.
        """
        # Initialize gradient accumulators for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Compute gradients for each training example in the mini-batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases by averaging the gradients
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Compute gradients for the cost function using backpropagation for a single training example.

        Parameters:
        - x (ndarray): Input data for a single training example.
        - y (ndarray): Corresponding target output for the training example.

        Returns:
        - tuple of lists: Gradients of biases and weights for each layer.
        """
        # Initialize gradients for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward pass: compute activations for each layer
        activation = x
        activations = [x]  # Store all activations, layer by layer
        zs = []  # Store all z vectors (weighted inputs), layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass: compute gradient of the cost function
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Compute gradients for remaining layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Calculate the accuracy of the network on the given data.

        Parameters:
        - data (list of tuples): List of tuples (x, y) where x is the input and y is the desired output.
        - convert (bool): Flag indicating whether the target output y should be converted to vectorized form.

        Returns:
        - float: Percentage of correctly classified samples.
        """
        # Compute the network's output for each example and compare it to the expected result
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        # Calculate the percentage of correctly classified samples
        return sum(int(x == y) for (x, y) in results) / len(data) * 100

    def cost_derivative(self, output_activations, y):
        """
        Compute the derivative of the cost function with respect to the output activations.

        Parameters:
        - output_activations (ndarray): Output activations from the network.
        - y (ndarray): Target output.

        Returns:
        - ndarray: Vector of partial derivatives ∂C_x / ∂a for the output activations.
        """
        # Difference between predicted and actual output
        return output_activations - y


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    - z (ndarray): Input value or array.

    Returns:
    - ndarray: Sigmoid of the input value or array.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function.

    Parameters:
    - z (ndarray): Input value or array.

    Returns:
    - ndarray: Derivative of the sigmoid function evaluated at the input value or array.
    """
    return sigmoid(z) * (1 - sigmoid(z))
