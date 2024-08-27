"""
mnist_avg_darkness.py
~~~~~~~~~~

This script demonstrates a baseline classifier using average darkness of images
for handwritten digit recognition on the MNIST dataset.
"""

from collections import defaultdict


def avg_darkness(training_data):
    """
    Compute average darkness for each digit from training data.

    Args:
    - training_data (tuple): A tuple containing training images and their corresponding labels.

    Returns:
    - averages (defaultdict): A dictionary mapping each digit (0-9) to its average darkness.

    Example:
    >>> averages = avg_darkness(training_data)
    """
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)

    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)

    averages = defaultdict(float)
    for digit, count in digit_counts.items():
        averages[digit] = darknesses[digit] / count

    return averages


def predict_digit(image, averages):
    """
    Predict the digit based on the closest average darkness to the given image.

    Args:
    - image (list): The image represented as a list of pixel values.
    - averages (defaultdict): A dictionary containing average darkness for each digit.

    Returns:
    - predicted_digit (int): The predicted digit based on closest average darkness.

    Example:
    >>> predicted_digit = predict_digit(image, averages)
    """
    darkness = sum(image)
    distances = {
        digit: abs(avg_darkness - darkness) for digit, avg_darkness in averages.items()
    }
    predicted_digit = min(distances, key=distances.get)

    return predicted_digit
