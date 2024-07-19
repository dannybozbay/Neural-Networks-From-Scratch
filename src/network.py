"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Standard library
import random

# Third-party libraries
import numpy as np

# Custom library
import mnist_loader


class Network:
    """
    A class representing a neural network for feedforward propagation and stochastic gradient descent training.

    Attributes:
    - sizes (list): List containing the number of neurons in each layer.
    - num_layers (int): Number of layers in the network.
    - biases ()

    Methods:
    - __init__(sizes): Initialize the network with given layer sizes.
    - feedforward(a): Perform feedforward propagation to compute activations.
    - SGD(training_data, epochs, mini_batch_size, eta, test_data=None):
        Train the network using mini-batch stochastic gradient descent.
    - update_mini_batch(mini_batch, eta): Update weights and biases using backpropagation for a mini-batch.
    - backprop(x, y): Compute gradients for the cost function using backpropagation.
    - evaluate(test_data): Evaluate the network's performance on test data.
    - cost_derivative(output_activations, y): Compute the derivative of the cost function.

    Example usage:
    >>> net = Network([784, 30, 10])
    >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    """

    def __init__(self, sizes):
        """
        Initialize the neural network with given layer sizes.

        Parameters:
        - sizes (list): List containing the number of neurons in each layer.

        Example:
        >>> net = Network([784, 30, 10])
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Perform feedforward propagation, computing activations using weights and biases.

        Parameters:
        - a (ndarray): Input activation vector for the network.

        Returns:
        - activations (ndarray): Output activation after propagation through the network.

        Example:
        >>> output = net.feedforward(input_data)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using stochastic gradient descent.

        Parameters:
        - training_data (list): List of tuples (x, y) representing training inputs and desired outputs.
        - epochs (int): Number of training epochs.
        - mini_batch_size (int): Size of mini-batches for stochastic gradient descent.
        - eta (float): Learning rate.
        - test_data (list, optional): If provided, the network will be evaluated against test data after each epoch.

        Example:
        >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.

        Parameters:
        - mini_batch (list): List of tuples (x, y) representing mini-batch inputs and desired outputs.
        - eta (float): Learning rate.

        Example:
        >>> net.update_mini_batch(mini_batch, eta)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return a tuple representing the gradient for the cost function C_x.

        Parameters:
        - x (ndarray): Input data for a single training example.
        - y (ndarray): Corresponding target output.

        Returns:
        - nabla_b (list): Gradient of biases for each layer.
        - nabla_w (list): Gradient of weights for each layer.

        Example:
        >>> nabla_b, nabla_w = net.backprop(x, y)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]  # List to store all the activations, layer by layer
        zs = []  # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Evaluate the network's performance on test data.

        Parameters:
        - test_data (list): List of tuples (x, y) representing test inputs and expected outputs.

        Returns:
        - int: Number of test inputs for which the network outputs the correct result.

        Example:
        >>> accuracy = net.evaluate(test_data)
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Compute the derivative of the cost function.

        Parameters:
        - output_activations (ndarray): Output activations from the network.
        - y (ndarray): Target output.

        Returns:
        - ndarray: Vector of partial derivatives ∂C_x / ∂a for the output activations.

        Example:
        >>> cost_derivative = net.cost_derivative(output_activations, y)
        """
        return output_activations - y


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    - z (ndarray): Input value or array.

    Returns:
    - ndarray: Sigmoid of the input value or array.

    Example:
    >>> sigmoid(np.array([0, 1, 2]))
    array([0.5       , 0.73105858, 0.88079708])
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function.

    Parameters:
    - z (ndarray): Input value or array.

    Returns:
    - ndarray: Derivative of the sigmoid function evaluated at the input value or array.

    Example:
    >>> sigmoid_prime(np.array([0, 1, 2]))
    array([0.25      , 0.19661193, 0.10499359])
    """
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == "__main__":
    train, validation, test = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test,
    )
