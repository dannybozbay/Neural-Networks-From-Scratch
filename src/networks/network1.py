"""
network1.py
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
    - __init__(sizes): Initializes the network with given layer sizes.
    - feedforward(a): Computes the output activations for a given input using feedforward propagation.
    - SGD(training_data, epochs, mini_batch_size, eta, validation_data=None, monitor_training_accuracy=False, monitor_validation_accuracy=False): Trains the network using mini-batch stochastic gradient descent.
    - update_mini_batch(mini_batch, eta): Updates weights and biases using backpropagation for a given mini-batch.
    - backprop(x, y): Computes gradients for the cost function using backpropagation for a single training example.
    - accuracy(data, convert=False): Calculates the accuracy of the network on the given data set.
    - cost_derivative(output_activations, y): Computes the derivative of the cost function with respect to output activations.
    """

    def __init__(self, sizes):
        """
        Initializes the neural network with given layer sizes and random weights and biases.

        Parameters:
        - sizes (list of int): List containing the number of neurons in each layer. The length of the list represents the number of layers in the network, including the input and output layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for all layers except the input layer with Gaussian distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights for all layers with Gaussian distribution
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Perform feedforward propagation for a single input vector.

        Parameters:
        - a (ndarray): Input activation vector, shape (input_size, 1).

        Returns:
        - ndarray: Output activation vector after propagation through the network, shape (output_size, 1).
        """
        # Iterate through each layer, updating the activations using the weights and biases
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(
                np.dot(w, a) + b
            )  # Compute the weighted input and apply the sigmoid activation function
        return a  # Return the final output activation vector

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
        Trains the network using mini-batch stochastic gradient descent.

        Parameters:
        - training_data (list of tuples): List of tuples (x, y) representing training inputs and target outputs.
        - epochs (int): Number of epochs for training.
        - mini_batch_size (int): Size of each mini-batch.
        - eta (float): Learning rate.
        - validation_data (list of tuples, optional): If provided, the network will be evaluated on this data after each epoch.
        - monitor_training_accuracy (bool, optional): If True, monitors accuracy on training data after each epoch.
        - monitor_validation_accuracy (bool, optional): If True, monitors accuracy on validation data after each epoch.

        Returns:
        - tuple: Two lists containing training accuracy and validation accuracy per epoch, if monitoring is enabled.
        """
        n = len(training_data)  # Total number of training samples
        training_acc, validation_acc = [], []  # Lists to store accuracy per epoch
        t0 = time.time()  # Start time for measuring training duration

        # Iterate over the number of epochs
        for j in range(1, epochs + 1):
            # Shuffle the training data to ensure randomness
            random.shuffle(training_data)

            # Create mini-batches from the shuffled training data
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Iterate over each mini-batch and update the network's weights and biases
            for mini_batch in mini_batches:
                # Perform gradient descent on the mini-batch
                self.update_mini_batch(mini_batch, eta)

            # Print progress of the current epoch and elapsed time
            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")
            INDENT = " " * 4  # Indentation for printing accuracy results

            # Monitor and print training accuracy if enabled
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_acc.append(accuracy)
                print(f"{INDENT}Accuracy on training data: {accuracy:.2f}%")

            # Monitor and print validation accuracy if enabled
            if monitor_validation_accuracy:
                accuracy = self.accuracy(validation_data)
                validation_acc.append(accuracy)
                print(f"{INDENT}Accuracy on validation data: {accuracy:.2f}%")

        return training_acc, validation_acc

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying stochastic gradient descent on a minibatch using backpropagation.

        Parameters:
        - mini_batch (list of tuples): List of tuples (x, y) representing mini-batch inputs and corresponding target outputs.
        - eta (float): Learning rate.
        """
        # Initialize gradient accumulators for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Compute gradients for each training example in the mini-batch using backpropagation
        # And add to the accumulators
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)  # Mini-batch size

        # Update weights and biases using the stochastic gradient descent rules
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Computes gradients for the cost function using backpropagation for a single training example.

        Parameters:
        - x (ndarray): Input data for a single training example, shape (input_size, 1).
        - y (ndarray): Corresponding target output for the training example, shape (output_size, 1).

        Returns:
        - tuple: Two lists containing gradients for biases and weights for each layer.
            - nabla_b (list of ndarray): Gradients for biases, each element has shape (neurons_in_layer, 1).
            - nabla_w (list of ndarray): Gradients for weights, each element has shape (neurons_in_next_layer, neurons_in_layer).
        """
        # Initialize gradients for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward pass: compute activations for each layer
        activation = x
        activations = [x]  # Store activations for each layer
        zs = []  # Store weighted inputs (z vectors) for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass: compute gradient of the cost function
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        Calculate the percentage of correct predictions made by the network on the given data.

        Parameters:
        - data (list of tuples): List of tuples (x, y) representing inputs and desired outputs.
        - convert (bool, optional): If True, indicates that the true output data (Y) is in vectorized form and needs to be converted to label format.

        Returns:
        - float: Percentage of correct predictions made by the network on the given data.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for x, y in data]

        return sum(int(x == y) for x, y in results) / len(data) * 100

    def cost_derivative(self, output_activations, y):
        """
        Compute the derivative of the cost function with respect to the output activations.

        Parameters:
        - output_activations (ndarray): Output activations from the network, shape (output_size, 1).
        - y (ndarray): True output values, shape (output_size, 1).

        Returns:
        - ndarray: Vector of derivative of the cost with respect to output activations, shape (output_size, 1).
        """
        return output_activations - y


# Utility functions


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
