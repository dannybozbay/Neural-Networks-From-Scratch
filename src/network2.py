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

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

import mnist_loader

plt.style.use(["science", "no-latex"])


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
        Calculate the error deltas from the output layers in a minibatch using for the quadratic cost function.

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
        Return the error delta from the output layer for the cross-entropy cost function.

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
        Initialize the neural network with given layer sizes.

        Parameters:
        - sizes (list): List containing the number of neurons in each layer.
        - cost (class): The cost function to be used (QuadraticCost or CrossEntropyCost).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        input connections to the neuron. Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1. Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison. It
        will usually be better to use the default weight initializer instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """
        Perform feedforward propagation, computing activations using weights and biases.

        Parameters:
        - a (ndarray): Input activation vector for the network.

        Returns:
        - ndarray: Output activation after propagation through the network.
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
        - training_data (list): List of tuples (x, y) representing training inputs and desired outputs.
        - epochs (int): Number of training epochs.
        - mini_batch_size (int): Size of mini-batches for stochastic gradient descent.
        - eta (float): Learning rate.
        - lmbda (float, optional): L2 regularization parameter (default is 0.0).
        - evaluation_data (list, optional): If provided, the network will be evaluated against the test data after each epoch.
        - monitor_evaluation_cost (bool, optional): Monitor cost on evaluation data (default is False).
        - monitor_evaluation_accuracy (bool, optional): Monitor accuracy on evaluation data (default is False).
        - monitor_training_cost (bool, optional): Monitor cost on training data (default is False).
        - monitor_training_accuracy (bool, optional): Monitor accuracy on training data (default is False).

        Returns:
        - tuple: Lists containing evaluation cost, evaluation accuracy, training cost, and training accuracy for each epoch.
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
        Update the network's weights and biases by applying gradient descent using backpropagation.

        Parameters:
        - mini_batch (list): List of tuples (x, y) representing mini-batch inputs and desired outputs.
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
        Perform backpropagation to compute gradients of the cost function.

        Parameters:
        - x (ndarray): Input data.
        - y (ndarray): Desired output.

        Returns:
        - tuple: Gradients of the biases and weights.
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
        - data (list): List of tuples (x, y) representing inputs and desired outputs.
        - convert (bool): Whether to convert y to one-hot representation.

        Returns:
        - int: Number of correct classifications.
        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost of the network on the provided data.

        Parameters:
        - data (list): List of tuples (x, y) representing inputs and desired outputs.
        - lmbda (float): L2 regularization parameter.
        - convert (bool): Whether to convert y to one-hot representation.

        Returns:
        - float: Total cost.
        """
        n = len(data)
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = mnist_loader.vectorized_result(y)
            cost += self.cost.fn(a, y) / n
        cost += 0.5 * (lmbda / n) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network to a file.

        Parameters:
        - filename (str): Name of the file to save the network.
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }
        with open(filename, "w") as f:
            json.dump(data, f)


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


if __name__ == "__main__":
    train, valid, test = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)

    # Train with large weights initializer
    net.large_weight_initializer()
    valid_cost1, valid_acc1, train_cost1, train_acc1 = net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Train with improved weights initializer (now called defaul_weight_initializer)
    net.default_weight_initializer()
    valid_cost2, valid_acc2, train_cost2, train_acc2 = net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Plot classification accuracies using large weighs initializer
    fig = plt.figure(figsize=(10, 6))
    plt.plot(valid_acc1, label="Validation data")
    plt.plot(train_acc1, label="Training data", c="orange")
    plt.title("Classification Accuracy (%) with Large Weights Initializer")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, len(valid_acc1) + 1, 5))
    plt.xlim(0, len(valid_acc1))
    plt.legend()
    plt.grid(True)
    plt.savefig("../reports/figures/accuracy_large_weights.png", dpi=1000)

    # Plot classification accuracies using improved weights initializer
    fig = plt.figure(figsize=(10, 6))
    plt.plot(valid_acc2, label="Validation data")
    plt.plot(train_acc2, label="Training data", c="orange")
    plt.title("Classification Accuracy (%) with Improved Weights Initializer")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, len(valid_acc2) + 1, 5))
    plt.xlim(0, len(valid_acc2))
    plt.legend()
    plt.grid(True)
    plt.savefig("../reports/figures/accuracy_improved_weights.png", dpi=1000)

    # Plot classification accuracies comparing original and improved weights initializer
    fig = plt.figure(figsize=(10, 6))
    plt.plot(valid_acc1, label="Original weights initialization")
    plt.plot(valid_acc2, label="Improved weights initialization", c="orange")
    plt.title("Classification Accuracy (%) On Evaluation Data")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, len(valid_acc1) + 1, 5))
    plt.xlim(0, len(valid_acc1))
    plt.legend()
    plt.grid(True)
    plt.savefig("../reports/figures/accuracy_compare_weights.png", dpi=1000)
