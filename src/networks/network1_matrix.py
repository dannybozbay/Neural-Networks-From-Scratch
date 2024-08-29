"""
network1_matrix.py
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
        Initialize the neural network with given layer sizes and random weights and biases.

        Parameters:
        - sizes (list of int): List containing the number of neurons in each layer. The length of the list represents the number of layers in the network, including the input and output layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for all layers except the input layer with Gaussian distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights for all layers with Gaussian distribution
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, A):
        """
        Perform feedforward propagation through the network using matrix operations.

        Parameters:
        - A (ndarray): Input activation matrix, shape (input_size, num_samples).

        Returns:
        - ndarray: Output activation matrix after propagation through the network, shape (output_size, num_samples).
        """
        m = A.shape[1]  # Number of samples

        # Iterate through each layer, updating the activations using the weights and biases
        for b, w in zip(self.biases, self.weights):
            B = np.tile(
                b, (1, m)
            )  # Broadcast the bias vector across all samples in the data
            A = sigmoid(
                np.dot(w, A) + B
            )  # Compute the weighted input and apply the sigmoid activation function

        return A  # Return the final output activation matrix

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
        n = len(training_data)  # Total number of training samples
        training_accuracy, validation_accuracy = (
            [],
            [],
        )  # Lists to store accuracy per epoch if monitoring
        t0 = time.time()  # Start time for measuring training duration

        # Iterate over the number of epochs
        for j in range(1, epochs + 1):
            random.shuffle(
                training_data
            )  # Shuffle the training data to ensure randomness

            # Create mini-batches from the shuffled training data
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Iterate over each mini-batch and update the network's weights and biases
            for mini_batch in mini_batches:
                X_minibatch = np.column_stack(
                    [x.ravel() for x, y in mini_batch]
                )  # Stack inputs
                Y_minibatch = np.column_stack(
                    [y.ravel() for x, y in mini_batch]
                )  # Stack outputs
                self.update_mini_batch(
                    X_minibatch, Y_minibatch, eta
                )  # Perform gradient descent on the mini-batch

            # Print progress of the current epoch and elapsed time
            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")

            INDENT = " " * 4  # Indentation for printing accuracy results

            # Monitor and print training accuracy if enabled
            if monitor_training_accuracy:
                # Stack the full training data to evaluate accuracy
                X_train = np.column_stack([x.ravel() for x, y in training_data])
                Y_train = np.column_stack([y.ravel() for x, y in training_data])
                accuracy = self.accuracy(
                    X_train, Y_train, convert=True
                )  # Compute training accuracy
                training_accuracy.append(
                    accuracy
                )  # Store training accuracy for this epoch
                print(f"{INDENT}Accuracy on training data: {accuracy:.2f}")

            # Monitor and print validation accuracy if enabled
            if monitor_validation_accuracy:
                # Stack the validation data to evaluate accuracy
                X_valid = np.column_stack([x.ravel() for x, y in validation_data])
                Y_valid = np.column_stack([y.ravel() for x, y in validation_data])
                accuracy = self.accuracy(
                    X_valid, Y_valid, convert=False
                )  # Compute validation accuracy
                validation_accuracy.append(
                    accuracy
                )  # Store validation accuracy for this epoch
                print(f"{INDENT}Accuracy on validation data: {accuracy:.2f}")

        return (
            training_accuracy,
            validation_accuracy,
        )  # Return the collected accuracy data

    def update_mini_batch(self, X, Y, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        Parameters:
        - X (ndarray): Input matrix for mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Target output matrix for mini-batch, shape (output_size, mini_batch_size).
        - eta (float): Learning rate.
        """
        # Ensure the number of samples in X and Y are equal
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )

        m = X.shape[1]  # Mini-batch size

        # Compute gradients for the entire mini-batch using backpropagation
        nabla_B, nabla_W = self.backprop(X, Y)

        # Update weights and biases, normalized by the mini-batch size
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_W)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_B)]

    def backprop(self, X, Y):
        """
        Compute gradients for the cost function using backpropagation for a mini-batch.

        Parameters:
        - X (ndarray): Input data for the mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Target outputs for the mini-batch, shape (output_size, mini_batch_size).

        Returns:
        - tuple: Two lists containing gradients for biases and weights for each layer.
        - nabla_B (list of ndarray): Gradients for biases, each element has shape (neurons_in_layer, 1).
        - nabla_W (list of ndarray): Gradients for weights, each element has shape (neurons_in_next_layer, neurons_in_layer).
        """
        # Ensure the number of samples in X and Y are equal
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )

        m = X.shape[1]  # Mini-batch size

        # Initialize gradient accumulators for biases and weights
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

        return nabla_B, nabla_W  # Return the accumulated gradients

    def accuracy(self, X, Y, convert=False):
        """
        Calculate the accuracy of the network on the given data.

        Parameters:
        - X (ndarray): Input data, shape (input_size, num_samples).
        - Y (ndarray): True output data, shape (output_size, num_samples) if convert is True; otherwise shape (1, num_samples) for labels.
        - convert (bool, optional): If True, indicates that the true output data (Y) is in vectorized form and needs to be converted to label format. Default is False.

        Returns:
        - float: Accuracy of the network in percentage.
        """
        # Ensure the number of samples in X and Y are equal
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )

        predictions = self.feedforward(X)  # Get network predictions

        # Convert network output to label format by selecting the index of the maximum value
        predictions = np.argmax(predictions, axis=0)

        if convert:
            # Convert true output data (Y) from vectorized form to label format
            Y = np.argmax(Y, axis=0)

        # Calculate accuracy by comparing predictions with true labels
        accuracy = np.mean(predictions == Y) * 100
        return accuracy

    def cost_derivative(self, output_activations, Y):
        """
        Compute the derivative of the cost function with respect to the output activations.

        Parameters:
        - output_activations (ndarray): Output activations from the network, shape (output_size, num_samples).
        - Y (ndarray): True output values, shape (output_size, num_samples).

        Returns:
        - ndarray: Derivative of the cost with respect to output activations, shape (output_size, num_samples).
        """
        return output_activations - Y  # Gradient of the quadratic cost function


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
