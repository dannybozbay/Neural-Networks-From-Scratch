"""
run_network2_matrix.py
~~~~~~~~~~~~~~~

"""

from data import mnist_loader
from networks import network2_matrix
from util.plots import plot_metrics

# Load and preprocess MNIST data
train, validation, test = mnist_loader.load_data_wrapper()

# Initialize the neural network with 784 input neurons, one hidden layer of 30 neurons, and 10 output neurons
net = network2_matrix.Network([784, 30, 10], cost=network2_matrix.CrossEntropyCost)

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
cost_plot.savefig("../reports/figures/network2_matrix/cost.png")
accuracy_plot.savefig("../reports/figures/network2_matrix/accuracy.png")
