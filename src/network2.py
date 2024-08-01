"""
network2.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. This implementation includes enhancements such
as the cross-entropy cost function, L2 regularization, and improved
initialization of network weights for better performance.

Note that the code focuses on simplicity, readability, and ease of
modification. It is not optimized and omits many desirable features.
"""

# Libraries
import json
import random
import sys
import time

import numpy as np

import mnist_loader


class QuadraticCost:
    """
    Quadratic cost function (Mean Squared Error).
    """

    @staticmethod
    def fn(a, y):
        """
        Calculate the quadratic cost associated with an output ``a`` and desired output ``y``.

        Parameters:
        - a (ndarray): Output activations from the network.
        - y (ndarray): Desired output activations.

        Returns:
        - float: Quadratic cost.
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        Calculate the error delta for the output layer using the quadratic cost function.

        Parameters:
        - z (ndarray): Weighted input to the output layer.
        - a (ndarray): Output activations from the network.
        - y (ndarray): Desired output activations.

        Returns:
        - ndarray: Error delta for the output layer.
        """
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    """
    Cross-entropy cost function.
    """

    @staticmethod
    def fn(a, y):
        """
        Calculate the cross-entropy cost associated with an output ``a`` and desired output ``y``.

        Parameters:
        - a (ndarray): Activation of the output layer (predicted output)
        - y (ndarray): Desired output.

        Returns:
        - float: The cross-entropy cost.
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Calculate the error delta for the output layer using the cross-entropy cost function.

        Parameters:
        - z (ndarray): Weighted input to the output layer.
        - a (ndarray): Output activations from the network.
        - y (ndarray): Desired output activations.

        Returns:
        - ndarray: Error delta for the output layer.
        """
        return a - y


class Network:
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

    def feedforward(self, a):
        """
        Perform feedforward propagation through the network.

        Parameters:
        - a (ndarray): Input activation vector.

        Returns:
        - ndarray: Output activation after passing through the network.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False,
    ):
        """
        Train the neural network using mini-batch stochastic gradient descent.

        Parameters:
        - training_data (list of tuples): List of tuples (x, y) representing training inputs and desired outputs.
        - epochs (int): Number of epochs for training.
        - mini_batch_size (int): Size of each mini-batch for stochastic gradient descent.
        - eta (float): Learning rate.
        - lmbda (float, optional): L2 regularization parameter (default is 0.0).
        - evaluation_data (list of tuples, optional): Data to evaluate the network after each epoch.
        - monitor_evaluation_cost (bool, optional): If True, monitor the cost on evaluation data (default is False).
        - monitor_evaluation_accuracy (bool, optional): If True, monitor the accuracy on evaluation data (default is False).
        - monitor_training_cost (bool, optional): If True, monitor the cost on training data (default is False).
        - monitor_training_accuracy (bool, optional): If True, monitor the accuracy on training data (default is False).

        Returns:
        - tuple: Four lists containing evaluation cost, evaluation accuracy, training cost, and training accuracy for each epoch.
        """
        if evaluation_data:
            n_eval = len(evaluation_data)
        n_train = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        t0 = time.time()  # Track the start time for epoch duration measurement
        for j in range(epochs):
            random.shuffle(training_data)  # Shuffle training data for each epoch
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n_train, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n_train)
            print(f"Epoch {j} complete (elapsed time: {round(time.time() - t0, 2)}s)")
            INDENT = " " * 4
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"{INDENT}Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"{INDENT}Accuracy on training data: {accuracy} / {n_train}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"{INDENT}Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"{INDENT}Accuracy on evaluation data: {accuracy} / {n_eval}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases using gradient descent on a mini-batch.

        Parameters:
        - mini_batch (list of tuples): List of tuples (x, y) representing mini-batch inputs and desired outputs.
        - eta (float): Learning rate.
        - lmbda (float): L2 regularization parameter.
        - n (int): Total number of training examples.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Perform backpropagation on a single training example to compute the gradient of the cost function.

        Parameters:
        - x (ndarray): Input activation vector.
        - y (ndarray): Desired output activation vector.

        Returns:
        - tuple: Gradients of biases and weights for each layer.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]  # Store all activations, layer by layer
        zs = []  # Store all z vectors (weighted inputs), layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Calculate the accuracy of the network on the provided data.

        Parameters:
        - data (list of tuples): List of tuples (x, y) representing inputs and desired outputs.
        - convert (bool): If True, convert `y` to one-hot representation (default is False).

        Returns:
        - int: Number of correct predictions.
        """
        results = (
            [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
            if convert
            else [(np.argmax(self.feedforward(x)), y) for x, y in data]
        )
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost of the network on the provided data, including L2 regularization.

        Parameters:
        - data (list of tuples): List of tuples (x, y) representing inputs and desired outputs.
        - lmbda (float): L2 regularization parameter.
        - convert (bool): If True, convert `y` to one-hot representation (default is False).

        Returns:
        - float: Total cost, including regularization term.
        """
        if convert:
            data = [(x, vectorized_result(y)) for x, y in data]
        # Propogate forward and calculate the total cost
        c0 = sum(self.cost.fn(self.feedforward(x), y) for x, y in data)
        # Add regularization term
        cost = c0 + (0.5 * lmbda * sum(np.sum(w**2) for w in self.weights))
        # Average by dividing by sample size
        return cost / len(data)

    def save(self, filename):
        """
        Save the network to a file.

        Parameters:
        - filename (str): Name of the file to save the network to.
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": self.cost.__name__,
        }
        with open(filename, "w") as f:
            json.dump(data, f)


def sigmoid(z):
    """
    Compute the sigmoid activation function.

    Parameters:
    - z (ndarray or float): Input to the sigmoid function.

    Returns:
    - ndarray or float: The sigmoid of the input.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid activation function.

    Parameters:
    - z (ndarray or float): Input to the sigmoid function.

    Returns:
    - ndarray or float: The derivative of the sigmoid function.
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def vectorized_result(j):
    """
    Convert a digit (0-9) into a corresponding 10-dimensional unit vector.

    Parameters:
    j (int): The digit to be converted into a one-hot encoded vector (0-9).

    Returns:
    numpy.ndarray: A 10-dimensional unit vector with a 1.0 in the jth position and 0.0 elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


#### Loading a Network
def load(filename):
    """
    Load a neural network from the file ``filename``.

    Parameters:
    - filename (str): Name of the file to load the network.

    Returns:
    - Network: An instance of the Network class.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
