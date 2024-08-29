"""
run_network1.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes a feedforward neural network from network1.py, and trains it using stochastic
gradient descent (SGD). It tracks the training and validation accuracy over multiple epochs and generates a plot of these metrics.

Steps:
1. Load and preprocess the MNIST dataset.
2. Set up a neural network with specified architecture.
3. Train the network using SGD with monitoring of accuracy.
4. Plot and save the training and validation accuracy.

Output:
- A PNG file showing the plot of training and validation accuracy over epochs, saved in reports/figures.
"""

from data import mnist_loader
from networks import network1
from util.plots import plot_metrics

# Load and preprocess MNIST data
train, validation, test = mnist_loader.load_data_wrapper()

# Initialize the neural network with 784 input neurons, one hidden layer of 30 neurons, and 10 output neurons
net = network1.Network([784, 30, 10])

# Train the network using stochastic gradient descent (SGD)
training_accuracy, validation_accuracy = net.SGD(
    training_data=train,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    validation_data=validation,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)

accuracy_plot = plot_metrics(
    [training_accuracy, validation_accuracy],
    ["Training Data", "Validation Data"],
    is_accuracy=True,
)

# Save the plot to the specified path
accuracy_plot.savefig(
    "../reports/figures/network1/accuracy_layers_784_30_10_eta_3_epochs_30.png"
)
