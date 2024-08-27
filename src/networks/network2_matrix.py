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

import numpy as np

# Custom library
import mnist_loader


class QuadraticCost:
    """
    Quadratic cost function (Mean Squared Error).
    """

    @staticmethod
    def fn(A, Y):
        """
        Calculate the quadratic costs associated with output matrix ``A`` and desired output matrix ``Y``.

        Parameters:
        - A (ndarray): Activations of the output layers. Shape: (output_size, batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, batch_size)

        Returns:
        - ndarray: A (1, batch_size) vector where each element is the quadratic cost for a single sample.
        """
        # Calculate the quadratic cost for each sample
        # Sum over rows (number of output neurons) to get the cost for each sample in the batch
        sample_costs = 0.5 * np.sum((A - Y) ** 2, axis=0, keepdims=True)
        return sample_costs

    @staticmethod
    def delta(Z, A, Y):
        """
        Calculate the error deltas for the output layers in a batch using the quadratic cost.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layers. Shape: (output_size, batch_size)
        - A (ndarray): Activations of the output layers. Shape: (output_size, batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, batch_size)

        Returns:
        - ndarray: Error deltas for the output layers. Shape: (output_size, batch_size)
        """
        return (A - Y) * sigmoid_prime(Z)


class CrossEntropyCost:
    """
    Cross-entropy cost function.
    """

    @staticmethod
    def fn(A, Y):
        """
        Calculate the cross-entropy costs associated with output matrix ``A`` and desired output matrix ``Y``.

        Parameters:
        - A (ndarray): Activations of the output layers. Shape: (output_size, batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, batch_size)

        Returns:
        - ndarray: A (1, batch_size) vector where each element is the cross-entropy cost for a single sample.
        """
        # Compute the cross-entropy cost for each element
        cost_matrix = -Y * np.log(A) - (1 - Y) * np.log(1 - A)
        # Sum over rows (number of output neurons) to get the cost for each sample in the batch
        sample_costs = np.sum(np.nan_to_num(cost_matrix), axis=0, keepdims=True)
        return sample_costs

    @staticmethod
    def delta(Z, A, Y):
        """
        Calculate the error deltas for the output layers in a batch using the cross-entropy cost.

        Parameters:
        - Z (ndarray): Weighted inputs to the output layers. Shape: (output_size, batch_size)
        - A (ndarray): Activations of the output layers. Shape: (output_size, batch_size)
        - Y (ndarray): Desired outputs. Shape: (output_size, batch_size)

        Returns:
        - ndarray: Error deltas for the output layers. Shape: (output_size, batch_size)
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
    - feedforward(A): Perform feedforward propagation to compute activations.
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

    def feedforward(self, A):
        """
        Perform feedforward propagation, computing activations using weights and biases.

        Parameters:
        - a (ndarray): Input activation vector for the network. Shape: (input_size, mini_batch_size)

        Returns:
        - ndarray: Output activation after propagation through the network. Shape: (output_size, mini_batch_size)
        """
        for b, w in zip(self.biases, self.weights):
            A = sigmoid(np.dot(w, A) + b)
        return A

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
        - mini_batch (list): List of tuples (x, y) for training.
        - eta (float): Learning rate.
        - lmbda (float): Regularization parameter.
        - n (int): Total size of the training set.
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
        Perform backpropagation to compute the gradient of the cost function with respect to weights and biases.

        Parameters:
        - X (ndarray): Input matrix of shape (input_size, mini_batch_size)
        - Y (ndarray): Output matrix of shape (output_size, mini_batch_size)

        Returns:
        - tuple: Gradients for biases and weights.
          nabla_b (list): Gradients for biases for each layer, each element has shape (neurons_in_layer, 1)
          nabla_w (list): Gradients for weights for each layer, each element has shape (neurons_in_next_layer, neurons_in_layer)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward pass
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
        - data (list): List of tuples (x, y) where x is the input and y is the desired output.
        - convert (bool): Flag to convert output to vectorized form.

        Returns:
        - int: Number of correctly classified samples.
        """
        results = (
            [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
            if convert
            else [(np.argmax(self.feedforward(x)), y) for x, y in data]
        )
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost for the given data set.

        Parameters:
        - data (list): List of tuples (x, y) representing the data inputs and desired outputs.
        - lmbda (float): Regularization parameter.
        - convert (bool): Flag to convert output to vectorized form.

        Returns:
        - float: Total cost for the given data set.
        """
        # Stack inputs horizontally. Shape: (input_size, data_size)
        X = np.column_stack([x.ravel() for x, y in data])
        # Stack outputs horizontally: Shape: (output_size, data_size)
        Y = np.column_stack([y.ravel() for x, y in data])
        if convert:
            Y = vectorized_result(Y)
        # Propogate forward
        A = self.feedforward(X)
        # Get the cost of each sample in data
        sample_costs = self.cost.fn(A, Y)
        # Calculate the total cost
        c0 = np.sum(sample_costs)
        # Add regularization term
        cost = c0 + (0.5 * lmbda * sum(np.linalg.norm(w) ** 2 for w in self.weights))
        # Average by dividing by sample size
        return cost / len(data)

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
    # Load training, validation, and test data using the custom mnist_loader
    train, validation, test = mnist_loader.load_data_wrapper()
    # Initialize the neural network with a specified structure
    net = Network([784, 30, 10])
    # Train the neural network using stochastic gradient descent
    net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test,
    )

    eval_cost, eval_acc, training_cost, training_acc = net.SGD(
        training_data=train,
        epochs=10,
        mini_batch_size=10,
        eta=0.1,
        lmbda=5.0,
        evaluation_data=validation,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_cost=True,
    )

    preds = net.feedforward(test)
