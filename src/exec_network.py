"""
exec_network.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes a feedforward neural network, and trains it using stochastic
gradient descent (SGD). It tracks the training and validation accuracy over multiple epochs and generates a plot of these metrics.

Steps:
1. Load and preprocess the MNIST dataset.
2. Set up a neural network with specified architecture.
3. Train the network using SGD with monitoring of accuracy.
4. Plot and save the training and validation accuracy.

Output:
- A PNG file showing the plot of training and validation accuracy over epochs.
"""

from data.mnist_loader import load_data_wrapper
from networks.network import Network
from util import *

# Load and preprocess MNIST data
train, validation, test = load_data_wrapper()

# Initialize the neural network with 784 input neurons, one hidden layer of 30 neurons, and 10 output neurons
net = Network([784, 30, 10])

# Train the network using stochastic gradient descent (SGD)
training_acc, validation_acc = net.SGD(
    training_data=train,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    validation_data=validation,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)

# Plot training and validation accuracy over epochs
plot = plot_metrics(training_acc, validation_acc)
plot.savefig(
    "../reports/figures/network_accuracy_layers_784_30_10_eta_3_epochs_30.png"
)  # Save the plot to the specified path
plt.show()  # Display the plot
