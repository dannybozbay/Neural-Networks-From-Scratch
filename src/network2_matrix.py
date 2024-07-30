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
import sys
import time

import matplotlib.pyplot as plt

# Libraries
import numpy as np
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"] = True

# Custom library
import mnist_loader


class QuadraticCost(object):
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
    def delta(Z, A, Y):
        """
        Calculate the error deltas for the output layers in a minibatch using the quadratic cost.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layers. Shape: (output_size, mini_batch_size)
        - A (ndarray): Activations of the output layers. Shape: (output_size, mini_batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, mini_batch_size)

        Returns:
        - ndarray: Error deltas for the output layers. Shape: (output_size, mini_batch_size)
        """
        return (A - Y) * sigmoid_prime(Z)


class CrossEntropyCost(object):
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
    def delta(Z, A, Y):
        """
        Calculate the error deltas for the output layers in a minibatch using the cross-entropy cost.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layers. Shape: (output_size, mini_batch_size)
        - A (ndarray): Activations of the output layers. Shape: (output_size, mini_batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, mini_batch_size)

        Returns:
        - ndarray: Error deltas for the output layer. Shape: (output_size, mini_batch_size)
        """
        return A - Y


class Network:
    """
    A class representing a neural network for feedforward propagation and stochastic gradient descent training.

    Attributes:
    - sizes (list): List containing the number of neurons in each layer.
    - num_layers (int): Number of layers in the network.
    - biases (list): List of bias vectors for each layer (excluding input layer).
    - weights (list): List of weight matrices for each layer.

    Methods:
    - __init__(sizes, cost=CrossEntropyCost): Initialize the network with given layer sizes.
    - default_weight_initializer(): Initialize weights and biases with a Gaussian distribution.
    - large_weight_initializer(): Initialize weights and biases with a standard Gaussian distribution.
    - feedforward(a): Perform feedforward propagation to compute activations.
    - SGD(training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data=None,
           monitor_evaluation_cost=False, monitor_evaluation_accuracy=False,
           monitor_training_cost=False, monitor_training_accuracy=False): Train the network using mini-batch stochastic gradient descent.
    - update_mini_batch(mini_batch, eta, lmbda, n): Update weights and biases using backpropagation for a mini-batch.
    - backprop(X, Y): Compute gradients for the cost function using backpropagation for a mini-batch.
    - accuracy(data, convert=False): Calculate the accuracy of the network on the given data.
    - total_cost(data, lmbda, convert=False): Calculate the total cost on the given data set.
    - save(filename): Save the neural network to a file.
    """

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initialize the neural network with given layer sizes.

        Parameters:
        - sizes (list): List containing the number of neurons in each layer.
        - cost (class): Cost function to be used (default is CrossEntropyCost).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Initialize weights and biases with a Gaussian distribution.
        Weights are scaled by the square root of the number of input connections.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """
        Initialize weights and biases with a standard Gaussian distribution.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """
        Perform feedforward propagation, computing activations using weights and biases.

        Parameters:
        - a (ndarray): Input activation vector for the network. Shape: (input_size, mini_batch_size)

        Returns:
        - ndarray: Output activation after propagation through the network. Shape: (output_size, mini_batch_size)
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
        - training_data (list): List of tuples (x, y) representing the training inputs and desired outputs.
        - epochs (int): Number of epochs to train for.
        - mini_batch_size (int): Size of each mini-batch.
        - eta (float): Learning rate.
        - lmbda (float): Regularization parameter.
        - evaluation_data (list): Data to evaluate the network performance, defaults to None.
        - monitor_evaluation_cost (bool): Flag to monitor evaluation cost.
        - monitor_evaluation_accuracy (bool): Flag to monitor evaluation accuracy.
        - monitor_training_cost (bool): Flag to monitor training cost.
        - monitor_training_accuracy (bool): Flag to monitor training accuracy.

        Returns:
        - tuple: Evaluation cost, evaluation accuracy, training cost, training accuracy.
        """
        if evaluation_data:
            n_eval = len(evaluation_data)
        n_train = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        t0 = time.time()
        for j in range(epochs):
            random.shuffle(training_data)
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
                print(f"{INDENT}Accuracy on training data: {accuracy}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"{INDENT}Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"{INDENT}Accuracy on evaluation data: {accuracy}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases using gradient descent on a single mini-batch.

        Parameters:
        - mini_batch (list): List of tuples (x, y) representing the mini-batch inputs and desired outputs.
        - eta (float): Learning rate.
        - lmbda (float): Regularization parameter.
        - n (int): Total size of the training data.
        """
        # Stack inputs horizontally
        X = np.column_stack(
            [x.ravel() for x, y in mini_batch]
        )  # Shape: (input_size, mini_batch_size)
        # Stack outputs horizontally
        Y = np.column_stack(
            [y.ravel() for x, y in mini_batch]
        )  # Shape: (output_size, mini_batch_size)

        # Compute gradients using backpropagation
        nabla_b, nabla_w = self.backprop(X, Y)

        # Update weights and biases using gradient descent
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, X, Y):
        """
        Compute gradients for the cost function using backpropagation for a mini-batch.

        Parameters:
        - X (ndarray): Input data for the mini-batch. Shape: (input_size, mini_batch_size)
        - Y (ndarray): Corresponding target outputs. Shape: (output_size, mini_batch_size)

        Returns:
        - tuple: Gradients of biases and weights for each layer.
          nabla_b (list): Gradients for biases, each element has shape (neurons_in_layer, 1)
          nabla_w (list): Gradients for weights, each element has shape (neurons_in_next_layer, neurons_in_layer)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = X
        activations = [X]  # List to store all activations, layer by layer
        Zs = []  # List to store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, activation) + b
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)

        # Backward pass
        delta = (self.cost).delta(Zs[-1], activations[-1], Y)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            Z = Zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Calculate the accuracy of the network on the given data.

        Parameters:
        - data (list): List of tuples (x, y) representing the inputs and desired outputs.
        - convert (bool): Flag to indicate if the desired output should be converted to a one-hot vector.

        Returns:
        - int: Number of correct predictions.
        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data) * 100

    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost on the given data set.

        Parameters:
        - data (list): List of tuples (x, y) representing the inputs and desired outputs.
        - lmbda (float): Regularization parameter.
        - convert (bool): Flag to indicate if the desired output should be converted to a one-hot vector.

        Returns:
        - float: Total cost on the data set.
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
        - filename (str): Name of the file to save the network to.
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
    Compute the sigmoid function.

    Parameters:
    - Z (ndarray): Input value or array.

    Returns:
    - ndarray: Sigmoid of the input value or array.
    """
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_prime(Z):
    """
    Compute the derivative of the sigmoid function.

    Parameters:
    - Z (ndarray): Input value or array.

    Returns:
    - ndarray: Derivative of the sigmoid function evaluated at the input value or array.
    """
    return sigmoid(Z) * (1 - sigmoid(Z))


def load(filename):
    """
    Load a neural network from a file.

    Parameters:
    - filename (str): Name of the file to load the network from.

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

    # Train WITHTOUT regularization
    net.large_weight_initializer()
    no_reg_valid_cost, no_reg_valid_acc, no_reg_train_cost, no_reg_train_acc = net.SGD(
        training_data=train,
        epochs=5,
        mini_batch_size=10,
        eta=0.5,
        lmbda=0.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Train WITH regularization (lambda=5.0)
    net.large_weight_initializer()
    reg_valid_cost, reg_valid_acc, reg_train_cost, reg_train_acc = net.SGD(
        training_data=train,
        epochs=5,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Plot accuracies on test set
    fig = plt.figure(figsize=(10, 6))
    plt.plot(no_reg_valid_acc, label=r"Without regularization: $\lambda = 0.0$")
    plt.plot(reg_valid_acc, label=r"With regularization: $\lambda = 5.0$", c="orange")
    plt.title(r"Classification Accuracy On Validation Data ($\eta = 0.5$)")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, len(reg_valid_acc) + 1, 5))
    plt.xlim(0, len(reg_valid_acc))
    plt.ylabel("%")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "../reports/figures/network2_matrix_compare_regularization.png", dpi=1000
    )

    # Train with large weights initializer
    net.large_weight_initializer()
    old_valid_cost, old_valid_acc, old_train_cost, old_train_acc = net.SGD(
        training_data=train,
        epochs=5,
        mini_batch_size=10,
        eta=0.1,
        lmbda=5.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Train with improved weights initializer
    net.default_weight_initializer()
    new_valid_cost, new_valid_acc, new_train_cost, new_train_acc = net.SGD(
        training_data=train,
        epochs=5,
        mini_batch_size=10,
        eta=0.1,
        lmbda=5.0,
        evaluation_data=valid,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
    )

    # Plot accuracies from old and new weights initializer
    fig = plt.figure(figsize=(10, 6))
    plt.plot(old_valid_acc, label="Old approach to weight initialization")
    plt.plot(
        new_valid_acc,
        label="New approach to weight initialization",
        c="orange",
    )
    plt.title(
        r"Classification Accuracy On Validation Data ($\eta = 0.1, \lambda = 5.0$)"
    )
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, len(old_valid_acc) + 1, 5))
    plt.xlim(0, len(old_valid_acc))
    plt.ylabel("%")
    plt.legend()
    plt.grid(True)
    plt.savefig("../reports/figures/network2_matrix_compare_weights.png", dpi=1000)
