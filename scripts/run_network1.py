"""
run_network1.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes a neural network with one hidden layer,
trains the network using stochastic gradient descent (SGD), and plots the training and validation accuracy.

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

# Plot accuracies
accuracy_plot = plot_metrics(
    [training_accuracy, validation_accuracy],
    ["Training Data", "Validation Data"],
    is_accuracy=True,
)

# Save plot
accuracy_plot.savefig(
    "../reports/figures/network1/accuracy_layers_784_30_10_eta_3_epochs_30.png"
)
