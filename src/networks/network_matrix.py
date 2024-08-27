"""
network_matrix.py
~~~~~~~~~~~~~~~~~

An optimized version of the feedforward neural network implementation with
stochastic gradient descent (SGD) learning algorithm. This version improves
training speed by utilizing matrix operations for gradient computation, rather
than iterating through individual training examples.

Key Features:
- Matrix-based feedforward propagation for efficient computation.
- Stochastic gradient descent for training with mini-batches.
- Backpropagation using matrix operations for gradient calculation.
- Performance improvements over the base implementation.

This code is designed to enhance training efficiency and is intended for
educational purposes and experimental use. It assumes that the training
data is well-prepared and does not include advanced features like regularization
or early stopping.
"""

# Libraries
import random
import time

import numpy as np


class Network:
    """
    A class representing a neural network with enhanced training speed through
    matrix-based operations for feedforward propagation and backpropagation.

    Attributes:
    - sizes (list of int): List containing the number of neurons in each layer.
    - num_layers (int): Total number of layers in the network.
    - biases (list of ndarray): List of bias vectors for each layer, excluding the input layer.
    - weights (list of ndarray): List of weight matrices for each layer.

    Methods:
    - __init__(sizes): Initialize the network with specified layer sizes.
    - feedforward(A): Perform feedforward propagation to compute activations for a given input matrix.
    - SGD(training_data, epochs, mini_batch_size, eta, validation_data=None, monitor_training_accuracy=False, monitor_validation_accuracy=False): Train the network using mini-batch stochastic gradient descent.
    - update_mini_batch(X, Y, eta): Update weights and biases using backpropagation for a mini-batch.
    - backprop(X, Y): Compute gradients for the cost function using backpropagation for a mini-batch.
    - accuracy(data, convert=False): Evaluate the network’s accuracy on given data.
    - cost_derivative(output_activations, Y): Compute the derivative of the cost function with respect to the output activations.
    """

    def __init__(self, sizes):
        """
        Initialize the neural network with given layer sizes.

        Parameters:
        - sizes (list of int): List containing the number of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for all layers except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights for all layers
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, A):
        """
        Perform feedforward propagation using matrix operations to compute activations.

        Parameters:
        - A (ndarray): Input activation matrix of shape (input_size, mini_batch_size).

        Returns:
        - ndarray: Output activation matrix after propagation through the network. Shape: (output_size, mini_batch_size).
        """
        m = A.shape[1]  # Number of samples in mini-batch
        for b, w in zip(self.biases, self.weights):
            B = np.tile(
                b, (1, m)
            )  # Broadcast bias vector across all samples in mini-batch
            A = sigmoid(np.dot(w, A) + B)
        return A

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
        Train the network using mini-batch stochastic gradient descent.

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
        n_train = len(training_data)
        training_acc, validation_acc = [], []
        t0 = time.time()  # Track the start time for epoch duration measurement
        for j in range(epochs):
            random.shuffle(
                training_data
            )  # Shuffle training data at the start of each epoch
            # Split training data into mini-batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)
            ]
            # Update the network for each mini-batch
            for mini_batch in mini_batches:
                X = np.column_stack([x.ravel() for x, y in mini_batch])
                Y = np.column_stack([y.ravel() for x, y in mini_batch])
                self.update_mini_batch(X, Y, eta)
            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")
            # Evaluate and print performance
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

    def update_mini_batch(self, X, Y, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        Parameters:
        - X (ndarray): Input matrix for mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Target output matrix for mini-batch, shape (output_size, mini_batch_size).
        - eta (float): Learning rate.
        """
        m = X.shape[1]  # Number of samples in mini-batch
        # Compute gradients for the entire mini-batch using backpropagation
        nabla_B, nabla_W = self.backprop(X, Y)
        # Update weights and biases by averaging the gradients
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_W)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_B)]

    def backprop(self, X, Y):
        """
        Compute gradients for the cost function using backpropagation for a mini-batch.

        Parameters:
        - X (ndarray): Input data for the mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Target outputs for the mini-batch, shape (output_size, mini_batch_size).

        Returns:
        - tuple: Gradients of biases and weights for each layer.
          - nabla_B (list of ndarray): Gradients for biases, each element has shape (neurons_in_layer, 1).
          - nabla_W (list of ndarray): Gradients for weights, each element has shape (neurons_in_next_layer, neurons_in_layer).
        """
        m = X.shape[1]  # Number of samples in mini-batch
        nabla_B = [np.zeros(b.shape) for b in self.biases]
        nabla_W = [np.zeros(w.shape) for w in self.weights]

        # Feedforward pass: compute activations for each layer
        activation = X
        activations = [X]  # Store activations for each layer
        Zs = []  # Store weighted inputs (z vectors) for each layer
        for b, w in zip(self.biases, self.weights):
            B = np.tile(b, (1, m))  # Broadcast bias vector across mini-batch
            Z = np.dot(w, activation) + B
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)

        # Backward pass: compute gradient of the cost function
        delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(Zs[-1])
        nabla_B[-1] = np.sum(
            delta, axis=1, keepdims=True
        )  # Sum over mini-batch dimension
        nabla_W[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            Z = Zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_B[-l] = np.sum(
                delta, axis=1, keepdims=True
            )  # Sum over mini-batch dimension
            nabla_W[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_B, nabla_W

    def accuracy(self, data, convert=False):
        """
        Calculate the accuracy of the network on the given data.

        Parameters:
        - data (list of tuples): List of tuples (x, y) where x is the input and y is the desired output.
        - convert (bool): Flag indicating whether the target output y should be converted to vectorized form.

        Returns:
        - float: Percentage of correctly classified samples.
        """
        # Stack inputs horizontally. Shape: (input_size, num_samples)
        X = np.column_stack([x.ravel() for x, y in data])
        # Stack outputs horizontally. Shape: (output_size, num_samples)
        Y = np.column_stack([y.ravel() for x, y in data])
        # Compute the network's output matrix. Shape: (output_size, num_samples)
        output = self.feedforward(X)

        if convert:
            # Convert network output and target output to class labels
            results = (np.argmax(output, axis=0), np.argmax(Y, axis=0))
        else:
            # Use raw output values and target values to determine correctness
            results = (np.argmax(output, axis=0), Y)

        # Calculate the percentage of correctly classified samples
        return np.sum(results[0] == results[1]) / len(data) * 100

    def cost_derivative(self, output_activations, Y):
        """
        Compute the derivative of the cost function with respect to the output activations.

        Parameters:
        - output_activations (ndarray): The output activations from the network.
        - Y (ndarray): The target output matrix.

        Returns:
        - ndarray: The derivative of the cost function with respect to the output activations.
        """
        return output_activations - Y


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.

    Parameters:
    - z (ndarray): Input array or matrix.

    Returns:
    - ndarray: The sigmoid of each element in the input array or matrix.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function for the input z.

    Parameters:
    - z (ndarray): Input array or matrix.

    Returns:
    - ndarray: The derivative of the sigmoid function for each element in the input array or matrix.
    """
    return sigmoid(z) * (1 - sigmoid(z))
