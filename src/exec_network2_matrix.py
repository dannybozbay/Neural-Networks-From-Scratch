import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network2 as n2
import network2_matrix as n2m
import plot_settings

train, validation, test = mnist_loader.load_data_wrapper()
net1 = n2.Network([784, 30, 10], cost=n2.CrossEntropyCost)
net2 = n2m.Network([784, 30, 10], cost=n2m.CrossEntropyCost)


eval_cost1, eval_acc1, training_cost1, training_acc1 = net1.SGD(
    training_data=train,
    epochs=10,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation,
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_cost=True,
)

eval_cost2, eval_acc2, training_cost2, training_acc2 = net2.SGD(
    training_data=train,
    epochs=10,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation,
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_cost=True,
)

# Plot accuracies
plt.plot(eval_acc1, label="Standard approach")
plt.plot(eval_acc2, label="Matrix approach")
plt.title(r"Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("%")
plt.xticks(np.arange(0, len(eval_acc1) + 1, 5))
plt.xlim(0, len(eval_acc1))
plt.legend()
plt.show()

# Plot costs
plt.plot(eval_cost1, label="Standard approach")
plt.plot(eval_cost2, label="Matrix approach")
plt.title(r"Cost")
plt.xlabel("Epoch")
plt.xticks(np.arange(0, len(eval_acc1) + 1, 5))
plt.xlim(0, len(eval_acc1))
plt.legend()
plt.show()
