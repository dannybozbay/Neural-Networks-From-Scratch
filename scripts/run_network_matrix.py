"""
run_network_matrix.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes a neural network with one hidden layer using matrix operations,
trains the network using stochastic gradient descent (SGD), and plots the training and validation accuracy.

"""

import sys

sys.path.append("..")

from src.data import mnist_loader
from src.networks.network_matrix import Network
from src.util.plots import *


def main():
    train, validation, test = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
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
    accuracy_plot.savefig(
        "../reports/figures/network_matrix/accuracy_layers_784_30_10_eta_3_epochs_30.png"
    )


if __name__ == "__main__":
    main()
