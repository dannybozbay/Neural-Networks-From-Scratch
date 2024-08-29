"""
run_network2.py
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
from networks import network2
from util.plots import plot_metrics

# Load and preprocess MNIST data
train, validation, test = mnist_loader.load_data_wrapper()

# Initialize the neural network with 784 input neurons, one hidden layer of 30 neurons, and 10 output neurons
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# Train the network using stochastic gradient descent (SGD)
training_cost, validation_cost, training_accuracy, validation_accuracy = net.SGD(
    training_data=train,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    validation_data=test,
    monitor_training_cost=True,
    monitor_validation_cost=True,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)

cost_plot = plot_metrics(
    [training_cost, validation_cost],
    ["Training Data", "Validation Data"],
    "Cross-Entropy Cost",
)

accuracy_plot = plot_metrics(
    [training_accuracy, validation_accuracy],
    ["Training Data", "Validation Data"],
    "Classification Accuracy",
    is_accuracy=True,
)

# Save the plots to the specified path
cost_plot.savefig("../reports/figures/network2/cost.png")
accuracy_plot.savefig("../reports/figures/network2/accuracy.png")
