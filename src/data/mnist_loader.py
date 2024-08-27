"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import gzip
import pickle

import numpy as np


def load_data():
    """
    Load the MNIST dataset from a gzip file and return it as a tuple.

    Returns:
    tuple: A tuple containing the training data, validation data, and test data.
        - training_data (tuple): Tuple with two entries:
            - numpy.ndarray: Contains 50,000 entries, each representing a 28x28 image (784 pixels).
            - numpy.ndarray: Contains 50,000 entries, each representing the digit (0-9) of the corresponding image.
        - validation_data (tuple): Tuple similar to training_data but with 10,000 images.
        - test_data (tuple): Tuple similar to validation_data but with 10,000 images.

    Notes:
    - The data is loaded from 'mnist.pkl.gz' file.
    - Each image is represented as a 1D numpy array with 784 values (28x28 pixels).
    """
    with gzip.open("../data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Return formatted data suitable for neural network training based on the MNIST dataset.

    Returns:
    tuple: A tuple containing formatted training, validation, and test data.
        - training_data (list): Contains 50,000 2-tuples (x, y):
            - x (numpy.ndarray): 784-dimensional array representing an input image.
            - y (numpy.ndarray): 10-dimensional unit vector representing the digit's correct classification.
        - validation_data (list): Contains 10,000 2-tuples (x, y):
            - x (numpy.ndarray): 784-dimensional array representing an input image.
            - y (int): Integer representing the digit classification (0-9).
        - test_data (list): Contains 10,000 2-tuples (x, y):
            - x (numpy.ndarray): 784-dimensional array representing an input image.
            - y (int): Integer representing the digit classification (0-9).

    Notes:
    - Uses the `load_data` function internally to load the MNIST dataset.
    - Formats training_data into a list of 2-tuples with inputs (x) reshaped to (784, 1) and outputs (y) vectorized.
    - Formats validation_data and test_data into lists of 2-tuples with inputs (x) reshaped to (784, 1) and outputs (y) as integers.
    """
    training_data, validation_data, test_data = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Convert a digit (0-9) into a corresponding 10-dimensional unit vector.

    Parameters:
    j (int): The digit to be converted into a one-hot encoded vector (0-9).

    Returns:
    numpy.ndarray: A 10-dimensional unit vector with a 1.0 in the jth position and 0.0 elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
