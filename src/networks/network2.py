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


class QuadraticCost:
    """
    Quadratic cost function (Mean Squared Error) for evaluating the performance of a neural network.
    """

    @staticmethod
    def fn(a, y):
        """
        Compute the quadratic cost for a single training example.

        This function calculates the quadratic cost between the predicted output activations and the desired output activations.

        Parameters:
        - a (ndarray): Output activation vector from the network, shape (output_size, 1).
        - y (ndarray): Desired output activation vector, shape (output_size, 1).

        Returns:
        - float: Quadratic cost, a scalar representing the error between the predicted and desired outputs.
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        Compute the error delta for the output layer based on the quadratic cost function.

        This function calculates the gradient of the cost function with respect to the output activations,
        which is used for backpropagation to update the network's weights and biases.

        Parameters:
        - z (ndarray): Weighted input to the output layer, shape (output_size, 1).
        - a (ndarray): Output activations from the network, shape (output_size, 1).
        - y (ndarray): Desired output activations, shape (output_size, 1).

        Returns:
        - ndarray: Error delta vector for the output layer, shape (output_size, 1).
        """
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    """
    Cross-entropy cost function for evaluating the performance of a neural network.
    """

    @staticmethod
    def fn(a, y):
        """
        Calculate the cross-entropy cost for a single output activation vector and the corresponding true label vector.

        This function computes the cross-entropy cost which quantifies the difference between the predicted
        output activations and the true labels for a single sample.

        Parameters:
        - a (ndarray): Output activation vector from the network, shape (output_size, 1). Represents the predicted probabilities.
        - y (ndarray): Desired output vector, shape (output_size, 1). Represents the true labels.

        Returns:
        - float: The cross-entropy cost for the given output activation and true label.
        """
        # To prevent log(0) issues, replace 0 values with a small number
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Calculate the error delta for the output layer using the cross-entropy cost function.

        This function computes the gradient of the cost function with respect to the output activations for a single sample,
        which is used to adjust the network's weights and biases during training.

        Parameters:
        - z (ndarray): Weighted input to the output layer, shape (output_size, 1). Not used in this calculation.
        - a (ndarray): Output activation vector from the network, shape (output_size, 1). Represents the predicted probabilities.
        - y (ndarray): Desired output vector, shape (output_size, 1). Represents the true labels.

        Returns:
        - ndarray: Error delta vector for the output layer, shape (output_size, 1). The gradient of the cost function with respect to the output activations.
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
        Perform feedforward propagation, computing activations using weights and biases.

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
        - evaluation_data (list of tuples, optional): Data to evaluate the network after each epoch.
        - monitor_evaluation_cost (bool, optional): If True, monitor the cost on evaluation data (default is False).
        - monitor_evaluation_accuracy (bool, optional): If True, monitor the accuracy on evaluation data (default is False).
        - monitor_training_cost (bool, optional): If True, monitor the cost on training data (default is False).
        - monitor_training_accuracy (bool, optional): If True, monitor the accuracy on training data (default is False).

        Returns:
        - tuple: Four lists containing evaluation cost, evaluation accuracy, training cost, and training accuracy for each epoch.
        """
        n = len(training_data)  # Total number of training samples
        training_cost, validation_cost = (
            [],
            [],
        )  # Lists to store cost per epoch if monitoring
        training_acc, validation_acc = (
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
                self.update_mini_batch(
                    mini_batch, eta, lmbda, n
                )  # Perform gradient descent on the mini-batch

            # Print progress of the current epoch and elapsed time
            print(f"Epoch {j} complete (elapsed time: {time.time() - t0:.2f}s)")

            INDENT = " " * 4  # Indentation for printing accuracy results

            # Monitor and print training cost if enabled
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)  # Compute training cost
                training_cost.append(cost)  # Store training cost for this epoch
                print(f"{INDENT}Cost on training data: {cost:.2f}")
            # Monitor and print validation cost if enabled
            if monitor_validation_cost:
                cost = self.total_cost(
                    validation_data, lmbda, convert=True
                )  # Compute validation cosr
                validation_cost.append(cost)  # Store validation cost for this epoch
                print(f"{INDENT}Cost on validation data: {cost:.2f}")
            # Monitor and print training accuracy if enabled
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_acc.append(accuracy)  # Compute training accuracy
                print(
                    f"{INDENT}Accuracy on training data: {accuracy:.2f}"
                )  # Store training accuracy for this epoch
            # Monitor and print validation accuracy if enabled
            if monitor_validation_accuracy:
                accuracy = self.accuracy(
                    validation_data, convert=False
                )  # Compute validation accuracy
                validation_acc.append(
                    accuracy
                )  # Store validation accuracy for this epoch
                print(f"{INDENT}Accuracy on validation data: {accuracy:.2f}")

        return (
            training_cost,
            validation_cost,
            training_acc,
            validation_acc,
        )  # Return the collected cost and accuracy data

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying stochastic gradient descent on a minibatch using backpropagation.

        Parameters:
        - mini_batch (list of tuples): List of tuples (x, y) representing mini-batch inputs and corresponding target outputs.
        - eta (float): Learning rate.
        - lmbda (float): L2 regularization parameter.
        - n (int): Total size of the training data set.
        """
        # Initialize gradient accumulators for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Compute gradients for each training example in the mini-batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)  # Mini-batch size

        # Update weights and biases using the stochastic gradient descent rules with regularization
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / m) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Compute gradients for the cost function using backpropagation for a single training example.

        Parameters:
        - x (ndarray): Input data for a single training example, shape (input_size, 1).
        - y (ndarray): Corresponding target output for the training example, shape (output_size, 1).

        Returns:
        - tuple: Two lists containing gradients for biases and weights for each layer.
        - nabla_B (list of ndarray): Gradients for biases, each element has shape (neurons_in_layer, 1).
        - nabla_W (list of ndarray): Gradients for weights, each element has shape (neurons_in_next_layer, neurons_in_layer).
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
        delta = self.cost.delta(zs[-1], activations[-1], y)
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
        Calculate the accuracy of the network on the given data.

        Parameters:
        - data (list of tuples): List of tuples (x, y) where x is the input and y is the desired output.
        - convert (bool, optional): If True, indicates that the true output data (Y) is in vectorized form and needs to be converted to label format. Default is False.

        Returns:
        - float: Accuracy of the network in percentage.
        """
        m = len(data)  # Number of samples
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        # Calculate the percentage of correctly classified samples
        return sum(int(x == y) for (x, y) in results) / m * 100

    def total_cost(self, data, lmbda, convert=False):
        """
        Calculate the total cost of the network on the given data, including L2 regularization.

        Parameters:
        - data (list of tuples): List of tuples (x, y) representing inputs and desired outputs.
        - lmbda (float): L2 regularization parameter.
        - convert (bool): If True, convert `y` to one-hot representation (default is False).

        Returns:
        - float: Total cost, including regularization term.
        """
        n = len(data)  # Number of sample
        # Convert true labels to one-hot vectors if flagged
        if convert:
            data = [(x, vectorized_result(y)) for x, y in data]
        # Propogate forward and calculate the ordinary total cost
        c0 = sum(self.cost.fn(self.feedforward(x), y) for x, y in data) / n
        # Compute and add regularization term
        cost = c0 + ((0.5 * lmbda / n) * sum(np.sum(w**2) for w in self.weights))
        return cost

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
