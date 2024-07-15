import matplotlib as plt
import numpy as np

from data.make_dataset import *

np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def feedforward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def feedforward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.feedforward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def feedforward(self, y_pred, y_true):
        epsilon = 1e-10
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        if len(y_true.shape) == 1:  # If target labels are not one-hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # If target labels are one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log_liklihoods = -np.log(correct_confidences)

        return neg_log_liklihoods


X, y = make_spiral_data(points=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

iterations = []
loss_values = []
acc_values = []

for iteration in range(100000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.feedforward(X)
    activation1.feedforward(dense1.output)
    dense2.feedforward(activation1.output)
    activation2.feedforward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(
            f"New set of weights found, iteration: {iteration}, loss: {loss}, acc: {accuracy}"
        )
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

        iterations.append(iteration)
        loss_values.append(loss)
        acc_values.append(accuracy)

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()


plt.subplot(1, 2, 1)
plt.plot(iterations, loss_values)
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(iterations, acc_values)
plt.title("Accuracy")

plt.show()
