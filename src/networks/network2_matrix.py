"""
network2_matrix.py
~~~~~~~~~~

A faster version of network2.py, implementing the stochastic gradient descent learning algorithm
for a feedforward neural network using a matrix-based approach for computing the gradients of mini-batches.

This approach processes inputs and outputs within a mini-batch as matrices, enabling efficient computation
using matrix operations and speeding up the training process.
"""

# Libraries
import json
import random
import time

import numpy as np


class QuadraticCost:
    """
    Quadratic cost function (Mean Squared Error) for evaluating the performance of a neural network with multiple training examples.
    """

    @staticmethod
    def fn(A, Y):
        """
        Calculate the quadratic cost for each sample in the output activation matrix.

        This function computes the quadratic cost between the predicted output activations and the true labels
        for each individual sample in a batch of training examples.

        Parameters:
        - A (ndarray): Output activation matrix from the network, shape (output_size, num_samples).
        - Y (ndarray): True label matrix, shape (output_size, num_samples).

        Returns:
        - ndarray: Row vector containing the quadratic cost for each sample, shape (1, num_samples).
        """
        # Ensure that A and Y have the same shape
        if A.shape != Y.shape:
            raise ValueError(
                f"Shape mismatch: A has shape {A.shape}, Y has shape {Y.shape}"
            )

        # Compute the element-wise squared difference between A and Y
        squared_diff = (A - Y) ** 2

        # Compute the quadratic cost for each training example
        cost_per_example = 0.5 * np.sum(
            squared_diff, axis=0, keepdims=True
        )  # Shape (1, num_samples)

        return cost_per_example

    @staticmethod
    def delta(Z, A, Y):
        """
        Calculate the error delta for the output layer for each sample in the output activation matrix.

        This function computes the gradient of the cost function with respect to the output activations for each sample,
        which is used to adjust the network's weights and biases during training.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layer, shape (output_size, num_samples).
        - A (ndarray): Output activations from the network, shape (output_size, num_samples).
        - Y (ndarray): True label matrix, shape (output_size, num_samples).

        Returns:
        - ndarray: Error delta matrix for the output layer, shape (output_size, num_samples).
        """
        return (A - Y) * sigmoid_prime(Z)


class CrossEntropyCost:
    """
    Cross-entropy cost function for evaluating the performance of a neural network.
    """

    @staticmethod
    def fn(A, Y):
        """
        Calculate the cross-entropy cost for each sample in the output activation matrix, A.

        This function computes the cross-entropy cost for each sample in a batch, where each column in A
        and Y represents a single training example. The result is a row vector containing the cross-entropy
        cost for each individual sample.

        Parameters:
        - A (ndarray): Output activation matrix from the network, shape (output_size, num_samples). Represents the predicted probabilities.
        - Y (ndarray): Desired output matrix, shape (output_size, num_samples). Represents the true labels.

        Returns:
        - ndarray: Row vector with the cross-entropy cost for each sample, shape (1, num_samples).
        """
        # Ensure that A and Y have the same shape
        if A.shape != Y.shape:
            raise ValueError(
                f"Shape mismatch: A has shape {A.shape}, Y has shape {Y.shape}"
            )

        # Compute the element-wise cross-entropy cost
        # Replace 0 values with a small number to avoid log(0)
        cross_entropy = np.nan_to_num(-Y * np.log(A) - (1 - Y) * np.log(1 - A))

        # Compute the total cost for each sample
        cost_per_example = np.sum(
            cross_entropy, axis=0, keepdims=True
        )  # Shape (1, num_samples)

        return cost_per_example

    @staticmethod
    def delta(Z, A, Y):
        """
        Calculate the error delta for the output layer for each sample in the output activation matrix.

        This function computes the gradient of the cost function with respect to the output activations for each sample,
        which is used to adjust the network's weights and biases during training.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layer, shape (output_size, num_samples).
        - A (ndarray): Output activations from the network, shape (output_size, num_samples).
        - Y (ndarray): True label matrix, shape (output_size, num_samples).

        Returns:
        - ndarray: Error delta matrix for the output layer, shape (output_size, num_samples).
        """
        return A - Y


class Network:
    """
    A feedforward neural network with support for various cost functions, L2 regularization,
    and different weight initialization strategies.
    """

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initialize the neural network with the given layer sizes and cost function.

        Parameters:
        - sizes (list of int): List containing the number of neurons in each layer.
        - cost (class): The cost function to be used (QuadraticCost or CrossEntropyCost).

        Attributes:
        - num_layers (int): Number of layers in the network.
        - sizes (list of int): List of layer sizes.
        - cost (class): Cost function used.
        - biases (list of ndarray): Biases for each layer.
        - weights (list of ndarray): Weights for each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Initialize weights and biases using a Gaussian distribution.

        Weights are initialized with mean 0 and standard deviation 1 over the square root of the number
        of input connections. Biases are initialized with mean 0 and standard deviation 1.

        Note:
        - The first layer is assumed to be an input layer and has no biases.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """
        Initialize weights and biases using a Gaussian distribution with mean 0 and standard deviation 1.

        Note:
        - The first layer is assumed to be an input layer and has no biases.
        - This initializer may not perform as well as the default initializer.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, A):
        """
        Perform feedforward propagation through the network using matrix operations.

        Parameters:
        - A (ndarray): Input activation matrix, shape (input_size, num_samples).

        Returns:
        - ndarray: Output activation matrix after propagation through the network, shape (output_size, num_samples).
        """
        m = A.shape[1]  # Number of samples
        for b, w in zip(self.biases, self.weights):
            B = np.tile(
                b, (1, m)
            )  # Broadcast the bias vector across all samples in the data
            A = sigmoid(np.dot(w, A) + B)
        return A

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        validation_data=None,
        monitor_training_cost=False,
        monitor_validation_cost=False,
        monitor_training_accuracy=False,
        monitor_validation_accuracy=False,
    ):
        """
        Train the network using mini-batch stochastic gradient descent.

        Parameters:
        - training_data (list of tuples): List of tuples (x, y) representing training inputs and desired outputs.
        - epochs (int): Number of epochs for training.
        - mini_batch_size (int): Size of each mini-batch for stochastic gradient descent.
        - eta (float): Learning rate.
        - lmbda (float, optional): L2 regularization parameter (default is 0.0).
        - validation_data (list of tuples, optional): Data to evaluate the network after each epoch.
        - monitor_training_cost (bool, optional): If True, monitor the cost on training data (default is False).
        - monitor_validation_cost (bool, optional): If True, monitor the cost on validation data (default is False).
        - monitor_training_accuracy (bool, optional): If True, monitor the accuracy on training data (default is False).
        - monitor_validation_accuracy (bool, optional): If True, monitor the accuracy on validation data (default is False).

        Returns:
        - tuple: Four lists containing training cost, validation cost, training accuracy, and validation accuracy for each epoch.
        """
        n = len(training_data)
        training_cost, validation_cost = [], []
        training_acc, validation_acc = [], []
        t0 = time.time()
        INDENT = " " * 4

        for j in range(1, epochs + 1):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                X_mini = np.column_stack([x.ravel() for x, y in mini_batch])
                Y_mini = np.column_stack([y.ravel() for x, y in mini_batch])
                self.update_mini_batch(X_mini, Y_mini, eta, lmbda, n)

            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")

            if monitor_training_cost or monitor_training_accuracy:
                X_train = np.column_stack([x.ravel() for x, y in training_data])
                Y_train = np.column_stack([y.ravel() for x, y in training_data])

            if monitor_validation_cost or monitor_validation_accuracy:
                X_valid = np.column_stack([x.ravel() for x, y in validation_data])
                Y_valid = np.column_stack([y.ravel() for x, y in validation_data])

            if monitor_training_cost:
                cost = self.total_cost(X_train, Y_train, lmbda)
                training_cost.append(cost)
                print(f"{INDENT}Cost on training data: {cost:.2f}")

            if monitor_validation_cost:
                cost = self.total_cost(X_valid, Y_valid, lmbda, convert=True)
                validation_cost.append(cost)
                print(f"{INDENT}Cost on validation data: {cost:.2f}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(X_train, Y_train, convert=True)
                training_acc.append(accuracy)
                print(f"{INDENT}Accuracy on training data: {accuracy:.2f}%")

            if monitor_validation_accuracy:
                accuracy = self.accuracy(X_valid, Y_valid, convert=False)
                validation_acc.append(accuracy)
                print(f"{INDENT}Accuracy on validation data: {accuracy:.2f}%")

        return training_cost, validation_cost, training_acc, validation_acc

    def update_mini_batch(self, X, Y, eta, lmbda, n):
        """
        Update the network's weights and biases by applying stochastic gradient descent on a minibatch.

        Parameters:
        - X (ndarray): Input matrix for the mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Output matrix for the mini-batch, shape (output_size, mini_batch_size).
        - eta (float): Learning rate.
        - lmbda (float): Regularization parameter.
        - n (int): Total number of training examples.
        """
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )

        m = X.shape[1]  # Mini-batch size
        nabla_b, nabla_w = self.backprop(X, Y)
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / m) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, X, Y):
        """
        Perform backpropagation to compute gradients of the cost function w.r.t the weights and biases
        for an entire mini-batchof data.

        Parameters:
        - X (ndarray): Input matrix for the mini-batch, shape (input_size, mini_batch_size).
        - Y (ndarray): Output matrix for the mini-batch, shape (output_size, mini_batch_size).

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
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

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
        delta = (self.cost).delta(Zs[-1], activations[-1], Y)
        nabla_b[-1] = np.sum(
            delta, axis=1, keepdims=True
        )  # Sum over mini-batch dimension
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            Z = Zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def accuracy(self, X, Y, convert=False):
        """
        Calculate the percentage of correct predictions made by the network on the given data.

        Parameters:
        X (ndarray): Input data, shape (input_size, num_samples).
        Y (ndarray): True output data, shape (output_size, num_samples) if convert is True; otherwise shape (1, num_samples) for labels.
        convert (bool, optional): If True, indicates that the true output data (Y) is in vectorized form and needs to be converted to label format. Default is False.

        Returns:
        float: Percentage of correct predictions made by the network on the given data.
        """
        # Ensure the number of samples in X and Y are equal
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )
        # Get the network's output activation matrix
        predictions = np.argmax(self.feedforward(X), axis=0)
        if convert:
            Y = np.argmax(Y, axis=0)

        return np.mean(predictions == Y) * 100

    def total_cost(self, X, Y, lmbda, convert=False):
        """
        Calculate the total cost of the network on the given data, including L2 regularization.

        Parameters:
        - data (list of tuples): List of tuples (x, y) representing inputs and desired outputs.
        - lmbda (float): L2 regularization parameter.
        - convert (bool): If True, convert `y` to one-hot representation (default is False).

        Returns:
        - float: Total cost, including regularization term.
        """
        # Ensure the number of samples in X and Y are equal
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Mismatch in number of samples: X has {X.shape[1]} samples, but Y has {Y.shape[1]} samples."
            )

        n = X.shape[1]  # Number of samples
        if convert:
            Y = vectorized_result(Y)
        # Propogate forward and compute the ordinary cost
        c0 = np.sum(self.cost.fn(self.feedforward(X), Y)) / n
        # Compute and add regularization term
        cost = c0 + ((0.5 * lmbda / n) * sum(np.sum(w**2) for w in self.weights))
        return cost

    def save(self, filename):
        """
        Save the neural network's architecture and parameters to a file.

        Parameters:
        - filename (str): Name of the file to save the network's parameters.
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }
        with open(filename, "w") as f:
            json.dump(data, f)


def sigmoid(Z):
    """
    Compute the sigmoid function for the input z.

    Parameters:
    - z (ndarray): Input value or matrix.

    Returns:
    - ndarray: Sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_prime(Z):
    """
    Compute the derivative of the sigmoid function for the input z.

    Parameters:
    - z (ndarray): Input value or matrix.

    Returns:
    - ndarray: Derivative of the sigmoid of z.
    """
    sig = sigmoid(Z)
    return sig * (1 - sig)


def vectorized_result(j):
    """
    Return a one-hot encoded matrix for the input array `j`.

    This function converts a given array of digits (0-9) into corresponding one-hot encoded vectors.
    If the input is a single column vector, the output will be a 10x1 vector with a 1.0 in the j-th position
    and zeroes elsewhere. If the input is a matrix with multiple columns, each column is independently
    converted to a one-hot encoded vector, resulting in a 10xm matrix.

    Parameters:
    - j (ndarray): An array or matrix of digits to be vectorized. If `j` is a column vector (shape (n, 1)),
                   it returns a 10x1 vector. If `j` is a matrix with multiple columns (shape (n, m)),
                   it returns a 10xm matrix.

    Returns:
    - ndarray: One-hot encoded matrix representing the input digits
    """
    if j.shape[1] == 1:
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
    else:
        m = j.shape[1]
        one_hot_matrix = np.zeros((10, m))
        one_hot_matrix[j, np.arange(m)] = 1
        return one_hot_matrix


def load(filename):
    """
    Load a neural network from a saved file.

    Parameters:
    - filename (str): Name of the file containing the network's parameters.

    Returns:
    - Network: A network instance with parameters loaded from the file.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    cost = globals()[data["cost"]]
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
