"""
run_network2_matrix.py
~~~~~~~~~~~~~~~

This script loads the MNIST dataset, initializes neural networks using different cost functions (quadratic and cross-entropy),
trains the networks with various configurations, and compares the results using plots.

"""

from data import mnist_loader
from networks import network2_matrix
from util.plots import plot_metrics

# Load and preprocess MNIST data
train, validation, test = mnist_loader.load_data_wrapper()

# Initialize a network with quadratic (MSE) cost
mse_net = network2_matrix.Network([784, 30, 10], cost=network2_matrix.QuadraticCost)
# Intialize a network with cross-entropy cost
ce_net = network2_matrix.Network([784, 30, 10], cost=network2_matrix.CrossEntropyCost)

# Set weights and biases using original initialization approach
mse_net.large_weight_initializer()
initial_weights, initial_biases = mse_net.weights, mse_net.biases
ce_net.weights, ce_net.biases = initial_weights, initial_biases

# Train both networks with no regularization
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

# Plots to compare MSE vs Cross-Entropy
cost_func_plot_acc = plot_metrics(
    [mse_eval_acc, ce_eval_acc],
    ["Quadratic Cost", "Cross-Entropy Cost"],
    "Classification Accuracy On The Validation Data",
    is_accuracy=True,
)
cost_func_plot_acc.savefig(
    "../reports/figures/network2_matrix/quadratic_vs_cross_entropy.png"
)

# Reinitialize the Cross-Entropy Network with intial params
ce_net.weights, ce_net.biases = initial_weights, initial_biases
# Train with regularization (lambda=5.0)
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

# Plot to compare regularization
reg_plot_acc = plot_metrics(
    [ce_eval_acc, ce_eval_acc2],
    [r"Regularization Off: $\lambda = 0.0$", r"Regularization On: $\lambda = 5.0$"],
    r"Classification Accuracy On The Validation Data",
    is_accuracy=True,
)
reg_plot_acc.savefig("../reports/figures/network2_matrix/acc_compare_reg.png")

# Reinitialize weights and biases of cross-entropy network using improved approach
ce_net.default_weight_initializer()
# Train with regularization (lambda=5.0)
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

# Plots to compare weight initialization approach
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
init_plot_acc.savefig("../reports/figures/network2_matrix/acc_compare_weight_init.png")

# Final plot to compare all improvements on accuracy
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
