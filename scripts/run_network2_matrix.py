"""
run_network2_matrix.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes neural networks using different cost functions (quadratic and cross-entropy),
trains the networks with various configurations, and compares the results using plots.

"""

import sys

sys.path.append("..")

from src.data import mnist_loader
from src.networks.network2_matrix import CrossEntropyCost, Network, QuadraticCost
from src.util.plots import *


def main():
    train, validation, test = mnist_loader.load_data_wrapper()
    mse_net = Network([784, 30, 10], cost=QuadraticCost)
    ce_net = Network([784, 30, 10], cost=CrossEntropyCost)
    mse_net.large_weight_initializer()
    initial_weights, initial_biases = mse_net.weights, mse_net.biases
    ce_net.weights, ce_net.biases = initial_weights, initial_biases

    mse_train_cost, mse_eval_cost, mse_train_acc, mse_eval_acc = mse_net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        lmbda=0,
        validation_data=validation,
        monitor_training_cost=True,
        monitor_validation_cost=True,
        monitor_training_accuracy=True,
        monitor_validation_accuracy=True,
    )

    ce_train_cost, ce_eval_cost, ce_train_acc, ce_eval_acc = ce_net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=3.0,
        lmbda=0,
        validation_data=validation,
        monitor_training_cost=True,
        monitor_validation_cost=True,
        monitor_training_accuracy=True,
        monitor_validation_accuracy=True,
    )

    cost_func_plot_acc = plot_metrics(
        [mse_eval_acc, ce_eval_acc],
        ["Quadratic Cost", "Cross-Entropy Cost"],
        "Classification Accuracy On The Validation Data",
        is_accuracy=True,
    )
    cost_func_plot_acc.savefig(
        "../reports/figures/network2_matrix/quadratic_vs_cross_entropy.png"
    )

    ce_net.weights, ce_net.biases = initial_weights, initial_biases
    ce_train_cost2, ce_eval_cost2, ce_train_acc2, ce_eval_acc2 = ce_net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        validation_data=validation,
        monitor_training_cost=True,
        monitor_validation_cost=True,
        monitor_training_accuracy=True,
        monitor_validation_accuracy=True,
    )

    reg_plot_acc = plot_metrics(
        [ce_eval_acc, ce_eval_acc2],
        [r"Regularization Off: $\lambda = 0.0$", r"Regularization On: $\lambda = 5.0$"],
        r"Classification Accuracy On The Validation Data",
        is_accuracy=True,
    )
    reg_plot_acc.savefig("../reports/figures/network2_matrix/acc_compare_reg.png")

    ce_net.default_weight_initializer()
    ce_train_cost3, ce_eval_cost3, ce_train_acc3, ce_eval_acc3 = ce_net.SGD(
        training_data=train,
        epochs=30,
        mini_batch_size=10,
        eta=0.5,
        lmbda=5.0,
        validation_data=validation,
        monitor_training_cost=True,
        monitor_validation_cost=True,
        monitor_training_accuracy=True,
        monitor_validation_accuracy=True,
    )

    init_plot_cost = plot_metrics(
        [ce_train_cost2, ce_train_cost3],
        ["Old weight initialization", "New weight initialization"],
        r"Cost on The Training Data ($\lambda = 5.0$)",
    )
    init_plot_cost.savefig(
        "../reports/figures/network2_matrix/cost_compare_weight_init.png"
    )
    init_plot_acc = plot_metrics(
        [ce_eval_acc2, ce_eval_acc3],
        ["Old weight initialization", "New weight initialization"],
        r"Classification Accuracy on The Validation Data ($\lambda = 5.0$)",
        is_accuracy=True,
    )
    init_plot_acc.savefig(
        "../reports/figures/network2_matrix/acc_compare_weight_init.png"
    )
    final_plot_acc = plot_metrics(
        [mse_eval_acc, ce_eval_acc, ce_eval_acc2, ce_eval_acc3],
        [
            "Quadratic Cost",
            "Cross-Entropy Cost",
            "Cross-Entropy + Regularization",
            "Cross-Entropy + Regularization + New Weight Initialization",
        ],
        "Classification Accuracy on The Validation Data",
        is_accuracy=True,
    )
    final_plot_acc.savefig("../reports/figures/network2_matrix/final_plot.png")


if __name__ == "__main__":
    main()
